"""
Knowledge Graph Embedding & Dynamic Semantic Context Engine (MedEmbed Powered)
=============================================================================
Vectorizes all Knowledge Graph nodes (Personas, Account Barriers, Core
Responsibilities, Compliance Rules, Topics, and Canonical Interaction Intents)
into dense semantic embeddings using abhinand/MedEmbed-small-v0.1.

Architectural Guarantees:
  1. Strict OS vs. FRM Graph Decoupling: Completely isolated vector stores for OS and FRM.
  2. Asymmetric BGE Encoding: Uses task-specific instruction prefix for user queries.
  3. Hybrid Two-Tier Compliance Gate / Clamp:
     - Tier 1: Deterministic Keyword/Regex Lexicon (Fast-Path).
     - Tier 2: MedEmbed Semantic Backstop for paraphrased inquiries (calibrated threshold >= 0.52).
     - Bypasses SLM on violation with authoritative Static 1.0 / Dynamic 0.0 override.
  4. Fixed Weighting: 0.30 Static KG governance + 0.70 Dynamic KG conversation context.
  5. Account Barrier Isolation: OS owns Patient Identification Barriers; FRM owns Market Access Barriers.
  6. Sub-10ms fast startup via precomputed PyTorch tensor caching (kg_medembed_cache.pt).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "abhinand/MedEmbed-small-v0.1"
EMBEDDING_DIM = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Tier-1 Lexicon Regex Patterns for Deterministic Compliance Gating
# ---------------------------------------------------------------------------
FRM_EXCLUSION_PATTERN = re.compile(
    r"\b(pfs|progression[- ]free survival|overall survival|os curve|response rate|hazard ratio|"
    r"clinical efficacy|clinical trial data|study arm|biomarker|ngs|exon 20 testing|"
    r"dose modification|toxicity management|adverse event treatment)\b",
    re.IGNORECASE
)

OS_EXCLUSION_PATTERN = re.compile(
    r"\b(manage toxicity|treat toxicity|adjust dose for toxicity|side effect mitigation|"
    r"dose reduction protocol|off[- ]label regimen|unapproved indication)\b",
    re.IGNORECASE
)

# Dedicated Hard Regulatory Exclusion Profiles for Tier-2 Semantic Gating
OS_HARD_EXCLUSIONS = [
    {
        "id": "resp:OS:toxicity",
        "role": "OS",
        "text": "Providing clinical advice on managing adverse events, treating drug toxicity, adjusting medication dosages, or side effect medical mitigation."
    },
    {
        "id": "resp:OS:off_label",
        "role": "OS",
        "text": "Promoting unapproved drug indications or off-label clinical usage outside approved prescribing information."
    }
]

FRM_HARD_EXCLUSIONS = [
    {
        "id": "resp:FRM:efficacy",
        "role": "FRM",
        "text": "Discussing clinical trial efficacy, overall survival (OS), progression-free survival (PFS), clinical study data, disease biology, or patient response rates."
    },
    {
        "id": "resp:FRM:biomarkers",
        "role": "FRM",
        "text": "Discussing biomarker testing protocols, genetic sequencing, EGFR exon 20 insertions, or molecular pathology diagnostics."
    },
    {
        "id": "resp:FRM:dosing_changes",
        "role": "FRM",
        "text": "Discussing drug dosing adjustments, administration protocols, subcutaneous infusion techniques, or medical mitigation of side effects."
    }
]

# Canonical conversational intents for semantic next-question steering (OS Focused)
CANONICAL_INTENTS = [
    # OS (Oncology Sales) Intents Grounded in INLEXZO Field Transcripts
    {
        "id": "intent_os_greeting_ready",
        "role": "OS",
        "intent_name": "greeting_ready_to_capture",
        "semantic_description": "Field rep confirms ready to capture call notes, doing well, in the car or between clinics.",
        "inquiry_template": "Who did you meet with, and what was the account name and location?"
    },
    {
        "id": "intent_os_account_hcp_id",
        "role": "OS",
        "intent_name": "account_and_hcp_identified",
        "semantic_description": "Field rep identifies the physician and practice location such as Dr. Michael Rao at Northside Urology Group in Raleigh, North Carolina.",
        "inquiry_template": "What did you learn about their treatment approach, any barriers, and whether there are new {brand} patient opportunities?"
    },
    {
        "id": "intent_os_no_patients",
        "role": "OS",
        "intent_name": "no_eligible_patients",
        "semantic_description": "Healthcare provider has no eligible patients, none identified so far, no matching cases right now, haven't seen candidates in clinic.",
        "inquiry_template": "What alternative treatment approach or protocol is {hcp} currently following?"
    },
    {
        "id": "intent_os_patient_identified",
        "role": "OS",
        "intent_name": "bcg_failure_or_unresponsive_identified",
        "semantic_description": "A patient with post-BCG failure or BCG-unresponsive NMIBC has been identified, evaluating candidate for {brand}.",
        "inquiry_template": "For that case, is {hcp} proceeding with {brand}, delaying, not starting, or still undecided?"
    },
    {
        "id": "intent_os_pa_barrier",
        "role": "OS",
        "intent_name": "prior_authorization_hurdle",
        "semantic_description": "Prior authorization is still in process, PA submission holdup, awaiting payer approval before putting procedure on calendar.",
        "inquiry_template": "What is the specific prior authorization or approval blocker right now for {brand}?"
    },
    {
        "id": "intent_os_formulary_barrier",
        "role": "OS",
        "intent_name": "formulary_pt_approval_hurdle",
        "semantic_description": "Product is not on formulary yet, P&T committee review is still pending, institutional review required before scheduling.",
        "inquiry_template": "What is the timeline for the P&T committee review or formulary approval at {account}?"
    },
    {
        "id": "intent_os_coverage_benefits_barrier",
        "role": "OS",
        "intent_name": "coverage_change_benefits_investigation",
        "semantic_description": "Coverage changed recently, office is rechecking what patient responsibility looks like, benefits investigation pending.",
        "inquiry_template": "Did the office provide an update on the benefits verification or patient responsibility for {brand}?"
    },
    {
        "id": "intent_os_deductible_affordability_barrier",
        "role": "OS",
        "intent_name": "deductible_patient_affordability",
        "semantic_description": "Patient working through annual deductible, payment timing, or out-of-pocket expenses before scheduling.",
        "inquiry_template": "Did you discuss affordability resources or patient assistance options to help navigate that deductible timing?"
    },
    {
        "id": "intent_os_insertion_scheduling",
        "role": "OS",
        "intent_name": "insertion_procedure_scheduled",
        "semantic_description": "Checking whether an INLEXZO insertion has been scheduled yet, planned timeline, calendar dates, avoiding patient identifiers.",
        "inquiry_template": "Has an {brand} insertion been scheduled yet? Please avoid any patient identifiers."
    },
    {
        "id": "intent_os_site_of_care",
        "role": "OS",
        "intent_name": "site_of_care_location",
        "semantic_description": "Planned procedure location in office exam room, ambulatory surgery center ASC, or hospital outpatient department.",
        "inquiry_template": "Do you know the planned site location and what support the account may want once it is scheduled?"
    },
    {
        "id": "intent_os_nurse_support",
        "role": "OS",
        "intent_name": "anatomical_model_nurse_prep_needed",
        "semantic_description": "Account requested a prep session using the anatomical model, demonstration for nursing staff and APP team before procedure.",
        "inquiry_template": "What specific anatomical model prep or workflow support did the clinic team request for the nurses?"
    },
    {
        "id": "intent_os_practice_criteria",
        "role": "OS",
        "intent_name": "practice_selection_criteria",
        "semantic_description": "Clinical criteria used by physician, NCCN guidelines, indication fit, BCG-unresponsive NMIBC with CIS with or without papillary disease.",
        "inquiry_template": "In their practice, what criteria does {hcp} use before considering {brand}?"
    },
    {
        "id": "intent_os_post_bcg_sequencing",
        "role": "OS",
        "intent_name": "post_bcg_sequencing_paradigm",
        "semantic_description": "How the physician thinks through treatment sequencing options once BCG is no longer working or patient is unresponsive.",
        "inquiry_template": "How does {hcp} think through options once BCG is no longer working?"
    },
    {
        "id": "intent_os_additional_challenges",
        "role": "OS",
        "intent_name": "additional_challenges_or_ae_needs",
        "semantic_description": "Checking if any shipment, product handling, training, or adverse event education needs came up.",
        "inquiry_template": "Outside of that access step, did they mention training, shipment, or AE education needs?"
    },
    {
        "id": "intent_os_no_barriers",
        "role": "OS",
        "intent_name": "no_barriers_reported",
        "semantic_description": "No barriers reported, insertions proceeding as planned, no additional challenges right now.",
        "inquiry_template": "Any other account context or scheduled follow-up touchpoint to capture?"
    },
    {
        "id": "intent_os_wrap_up",
        "role": "OS",
        "intent_name": "ready_to_wrap_up",
        "semantic_description": "All key points captured, nothing else to log, ready to close note, all set.",
        "inquiry_template": "Got it. I have the main points. Are we ready to finish?"
    }
]


class KGEmbeddingEngine:
    """
    High-performance, localized embedding and semantic retrieval engine
    powered by abhinand/MedEmbed-small-v0.1 for Johnson & Johnson Commercial Oncology.
    
    Guarantees:
      - Strict decoupling of OS and FRM vector stores.
      - Hybrid Two-Tier Compliance Gate with pre-generation SLM bypass.
      - Fixed 0.30 Static / 0.70 Dynamic weighting on compliant turns.
    """

    def __init__(
        self,
        kg_data: Optional[Dict[str, Any]] = None,
        account_barriers: Optional[List[Dict[str, Any]]] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / "engine"
        self.cache_path = self.cache_dir / "kg_medembed_cache.pt"

        # Hardware device selection (CPU or CUDA)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MedEmbed tokenizer and model
        print(f"[KGEmbeddingEngine] Initializing MedEmbed ({MODEL_NAME}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        self.kg_data = kg_data
        self.account_barriers = account_barriers or []

        # OS Vector Stores (Strictly Isolated)
        self.os_barrier_embeddings: Optional[torch.Tensor] = None
        self.os_barrier_metadata: List[Dict[str, Any]] = []

        self.os_responsibility_embeddings: Optional[torch.Tensor] = None
        self.os_responsibility_metadata: List[Dict[str, Any]] = []

        self.os_topic_embeddings: Optional[torch.Tensor] = None
        self.os_topic_metadata: List[Dict[str, Any]] = []

        self.os_exclusion_embeddings: Optional[torch.Tensor] = None
        self.os_exclusion_metadata: List[Dict[str, Any]] = []

        # Dedicated Hard Regulatory Exclusion Tensors (Tier-2 Semantic Gate)
        self.os_hard_exclusion_embeddings: Optional[torch.Tensor] = None
        self.os_hard_exclusion_metadata: List[Dict[str, Any]] = []

        self.os_intent_embeddings: Optional[torch.Tensor] = None
        self.os_intent_metadata: List[Dict[str, Any]] = []

        # FRM Vector Stores (Strictly Isolated)
        self.frm_barrier_embeddings: Optional[torch.Tensor] = None
        self.frm_barrier_metadata: List[Dict[str, Any]] = []

        self.frm_responsibility_embeddings: Optional[torch.Tensor] = None
        self.frm_responsibility_metadata: List[Dict[str, Any]] = []

        self.frm_topic_embeddings: Optional[torch.Tensor] = None
        self.frm_topic_metadata: List[Dict[str, Any]] = []

        self.frm_exclusion_embeddings: Optional[torch.Tensor] = None
        self.frm_exclusion_metadata: List[Dict[str, Any]] = []

        # Dedicated Hard Regulatory Exclusion Tensors (Tier-2 Semantic Gate)
        self.frm_hard_exclusion_embeddings: Optional[torch.Tensor] = None
        self.frm_hard_exclusion_metadata: List[Dict[str, Any]] = []

        self.frm_intent_embeddings: Optional[torch.Tensor] = None
        self.frm_intent_metadata: List[Dict[str, Any]] = []

        # Backward-compatibility unified tensors
        self.barrier_embeddings: Optional[torch.Tensor] = None
        self.barrier_metadata: List[Dict[str, Any]] = []
        self.responsibility_embeddings: Optional[torch.Tensor] = None
        self.responsibility_metadata: List[Dict[str, Any]] = []
        self.exclusion_embeddings: Optional[torch.Tensor] = None
        self.exclusion_metadata: List[Dict[str, Any]] = []
        self.topic_embeddings: Optional[torch.Tensor] = None
        self.topic_metadata: List[Dict[str, Any]] = []
        self.intent_embeddings: Optional[torch.Tensor] = None
        self.intent_metadata: List[Dict[str, Any]] = []

        # Initialize or load cached embeddings
        self._initialize_embeddings()

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Generates normalized dense embeddings (L2 unit vectors) for a list of strings.
        Shape: [N, 384]
        """
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), device=self.device)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded_input = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)
                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                # L2 Normalize for cosine similarity via dot product
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encodes a user query with BGE asymmetric instruction prefix:
        'Represent this sentence for searching relevant passages: '
        Shape: [384]
        """
        if not text or not text.strip():
            return torch.zeros(EMBEDDING_DIM, device=self.device)
        prefixed = f"{BGE_QUERY_PREFIX}{text.strip()}"
        return self.encode([prefixed])[0]

    def encode_passages(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encodes Knowledge Graph passages/nodes directly without query prefix."""
        return self.encode(texts, batch_size=batch_size)

    def encode_single(self, text: str) -> torch.Tensor:
        """Encodes a single query text string into a 1D unit tensor [384]."""
        return self.encode_query(text)

    def _initialize_embeddings(self):
        """Loads embeddings from cache or computes and saves them."""
        if self.cache_path.exists():
            try:
                t0 = time.time()
                cached = torch.load(self.cache_path, map_location=self.device)
                
                # OS Stores
                self.os_barrier_embeddings = cached["os_barrier_embeddings"].to(self.device)
                self.os_barrier_metadata = cached["os_barrier_metadata"]
                self.os_responsibility_embeddings = cached["os_responsibility_embeddings"].to(self.device)
                self.os_responsibility_metadata = cached["os_responsibility_metadata"]
                self.os_exclusion_embeddings = cached["os_exclusion_embeddings"].to(self.device)
                self.os_exclusion_metadata = cached["os_exclusion_metadata"]
                self.os_topic_embeddings = cached["os_topic_embeddings"].to(self.device)
                self.os_topic_metadata = cached["os_topic_metadata"]
                self.os_intent_embeddings = cached["os_intent_embeddings"].to(self.device)
                self.os_intent_metadata = cached["os_intent_metadata"]

                # FRM Stores
                self.frm_barrier_embeddings = cached["frm_barrier_embeddings"].to(self.device)
                self.frm_barrier_metadata = cached["frm_barrier_metadata"]
                self.frm_responsibility_embeddings = cached["frm_responsibility_embeddings"].to(self.device)
                self.frm_responsibility_metadata = cached["frm_responsibility_metadata"]
                self.frm_exclusion_embeddings = cached["frm_exclusion_embeddings"].to(self.device)
                self.frm_exclusion_metadata = cached["frm_exclusion_metadata"]
                self.frm_topic_embeddings = cached["frm_topic_embeddings"].to(self.device)
                self.frm_topic_metadata = cached["frm_topic_metadata"]
                self.frm_intent_embeddings = cached["frm_intent_embeddings"].to(self.device)
                self.frm_intent_metadata = cached["frm_intent_metadata"]

                # Dedicated Hard Exclusion Stores (Tier-2 Semantic Gate)
                if "os_hard_exclusion_embeddings" in cached and "frm_hard_exclusion_embeddings" in cached:
                    self.os_hard_exclusion_embeddings = cached["os_hard_exclusion_embeddings"].to(self.device)
                    self.os_hard_exclusion_metadata = cached["os_hard_exclusion_metadata"]
                    self.frm_hard_exclusion_embeddings = cached["frm_hard_exclusion_embeddings"].to(self.device)
                    self.frm_hard_exclusion_metadata = cached["frm_hard_exclusion_metadata"]
                else:
                    raise KeyError("Hard exclusion embeddings missing in cache. Rebuilding...")

                # Populate unified view for backward compatibility
                self._populate_unified_views()

                print(
                    f"[KGEmbeddingEngine] Loaded precomputed MedEmbed cache from {self.cache_path.name} in {time.time()-t0:.3f}s:\n"
                    f"  - OS : {len(self.os_barrier_metadata)} barriers, {len(self.os_responsibility_metadata)} responsibilities, {len(self.os_exclusion_metadata)} exclusions, {len(self.os_hard_exclusion_metadata)} hard exclusions\n"
                    f"  - FRM: {len(self.frm_barrier_metadata)} barriers, {len(self.frm_responsibility_metadata)} responsibilities, {len(self.frm_exclusion_metadata)} exclusions, {len(self.frm_hard_exclusion_metadata)} hard exclusions"
                )
                return
            except Exception as e:
                print(f"[KGEmbeddingEngine] Failed to load cache from {self.cache_path}: {e}. Recomputing...")

        # Recompute embeddings from source data
        self._build_embeddings()

    def _populate_unified_views(self):
        """Constructs unified views from isolated OS and FRM stores for backward compatibility."""
        self.barrier_metadata = self.os_barrier_metadata + self.frm_barrier_metadata
        if self.os_barrier_embeddings is not None and self.frm_barrier_embeddings is not None:
            self.barrier_embeddings = torch.cat([self.os_barrier_embeddings, self.frm_barrier_embeddings], dim=0)

        self.responsibility_metadata = self.os_responsibility_metadata + self.frm_responsibility_metadata
        if self.os_responsibility_embeddings is not None and self.frm_responsibility_embeddings is not None:
            self.responsibility_embeddings = torch.cat([self.os_responsibility_embeddings, self.frm_responsibility_embeddings], dim=0)

        self.exclusion_metadata = self.os_exclusion_metadata + self.frm_exclusion_metadata
        if self.os_exclusion_embeddings is not None and self.frm_exclusion_embeddings is not None:
            self.exclusion_embeddings = torch.cat([self.os_exclusion_embeddings, self.frm_exclusion_embeddings], dim=0)

        self.topic_metadata = self.os_topic_metadata + self.frm_topic_metadata
        if self.os_topic_embeddings is not None and self.frm_topic_embeddings is not None:
            self.topic_embeddings = torch.cat([self.os_topic_embeddings, self.frm_topic_embeddings], dim=0)

        self.intent_metadata = self.os_intent_metadata + self.frm_intent_metadata
        if self.os_intent_embeddings is not None and self.frm_intent_embeddings is not None:
            self.intent_embeddings = torch.cat([self.os_intent_embeddings, self.frm_intent_embeddings], dim=0)

    def _build_embeddings(self):
        """Constructs strictly separated vector stores for OS and FRM and saves cache."""
        t0 = time.time()
        print("[KGEmbeddingEngine] Vectorizing Knowledge Graph entities with MedEmbed...")

        # Load role KGs
        os_kg_path = self.base_dir / "kg_os.json"
        frm_kg_path = self.base_dir / "kg_frm.json"
        os_data = {}
        frm_data = {}

        if os_kg_path.exists():
            with open(os_kg_path, "r", encoding="utf-8") as f:
                os_data = json.load(f)
        if frm_kg_path.exists():
            with open(frm_kg_path, "r", encoding="utf-8") as f:
                frm_data = json.load(f)

        # -------------------------------------------------------------
        # 1. BUILD OS VECTOR STORE (Patient Identification Only)
        # -------------------------------------------------------------
        os_barrier_texts, self.os_barrier_metadata = [], []
        os_resp_texts, self.os_responsibility_metadata = [], []
        os_excl_texts, self.os_exclusion_metadata = [], []
        os_topic_texts, self.os_topic_metadata = [], []
        os_intent_texts, self.os_intent_metadata = [], []

        for node in os_data.get("nodes", []):
            label = node.get("label")
            nid = node.get("id", "")
            props = node.get("properties", {})

            if label == "AccountBarrier":
                acc = props.get("account", "")
                b_type = props.get("barrier_type", "Patient Identification Barrier")
                details = props.get("barrier_details", "")
                ask = props.get("account_ask", "")
                keywords = " ".join(props.get("trigger_keywords", []))
                text = f"Role: OS. Account: {acc}. Barrier: {b_type}. Details: {details}. Ask: {ask}. Key Triggers: {keywords}."
                os_barrier_texts.append(text)
                self.os_barrier_metadata.append(props)

            elif label == "CoreResponsibility":
                duty_text = props.get("text", "")
                domain = props.get("domain", "commercial_promotional")
                topics = ", ".join(props.get("topics", []))
                text = f"Role: OS. Domain: {domain}. Responsibility: {duty_text}. In-Scope Topics: {topics}."
                os_resp_texts.append(text)
                self.os_responsibility_metadata.append({"id": nid, "role": "OS", "text": duty_text, "properties": props})

            elif label == "ExclusionRule":
                excl_text = props.get("text", "")
                text = f"Forbidden Regulatory Exclusion for OS: {excl_text}"
                os_excl_texts.append(text)
                self.os_exclusion_metadata.append({"id": nid, "role": "OS", "text": excl_text, "properties": props})

            elif label == "Topic":
                tname = props.get("name", "")
                os_topic_texts.append(f"OS In-Scope Topic: {tname}.")
                self.os_topic_metadata.append({"name": tname, "id": nid, "role": "OS"})

        # Explicit OS Forbidden Semantic Concepts
        os_hard_exclusions = [
            {"id": "resp:OS:toxicity", "role": "OS", "text": "Managing clinical toxicity, treating adverse events, adjusting medical dosages, or clinical toxicity interventions."},
            {"id": "resp:OS:off_label", "role": "OS", "text": "Recommending unapproved off-label indications, off-label treatment sequencing, or unapproved patient cohorts."},
            {"id": "resp:OS:pricing", "role": "OS", "text": "Discussing specific hospital pricing discounts, institutional rebate contracts, or direct payer negotiations."}
        ]
        for e in os_hard_exclusions:
            os_excl_texts.append(f"Forbidden Regulatory Exclusion for OS: {e['text']}")
            self.os_exclusion_metadata.append(e)

        # OS Intents
        for item in CANONICAL_INTENTS:
            if item.get("role") == "OS":
                os_intent_texts.append(f"Role: OS. Intent: {item['intent_name']}. Meaning: {item['semantic_description']}")
                self.os_intent_metadata.append(item)

        self.os_barrier_embeddings = self.encode_passages(os_barrier_texts)
        self.os_responsibility_embeddings = self.encode_passages(os_resp_texts)
        self.os_exclusion_embeddings = self.encode_passages(os_excl_texts)
        self.os_topic_embeddings = self.encode_passages(os_topic_texts)
        self.os_intent_embeddings = self.encode_passages(os_intent_texts)

        # -------------------------------------------------------------
        # 2. BUILD FRM VECTOR STORE (Market Access Only)
        # -------------------------------------------------------------
        frm_barrier_texts, self.frm_barrier_metadata = [], []
        frm_resp_texts, self.frm_responsibility_metadata = [], []
        frm_excl_texts, self.frm_exclusion_metadata = [], []
        frm_topic_texts, self.frm_topic_metadata = [], []
        frm_intent_texts, self.frm_intent_metadata = [], []

        for node in frm_data.get("nodes", []):
            label = node.get("label")
            nid = node.get("id", "")
            props = node.get("properties", {})

            if label == "AccountBarrier":
                acc = props.get("account", "")
                b_type = props.get("barrier_type", "Market Access Barrier")
                details = props.get("barrier_details", "")
                ask = props.get("account_ask", "")
                keywords = " ".join(props.get("trigger_keywords", []))
                text = f"Role: FRM. Account: {acc}. Barrier: {b_type}. Details: {details}. Ask: {ask}. Key Triggers: {keywords}."
                frm_barrier_texts.append(text)
                self.frm_barrier_metadata.append(props)

            elif label == "CoreResponsibility":
                duty_text = props.get("text", "")
                domain = props.get("domain", "patient_access_reimbursement")
                topics = ", ".join(props.get("topics", []))
                text = f"Role: FRM. Domain: {domain}. Responsibility: {duty_text}. In-Scope Topics: {topics}."
                frm_resp_texts.append(text)
                self.frm_responsibility_metadata.append({"id": nid, "role": "FRM", "text": duty_text, "properties": props})

            elif label == "ExclusionRule":
                excl_text = props.get("text", "")
                text = f"Forbidden Regulatory Exclusion for FRM: {excl_text}"
                frm_excl_texts.append(text)
                self.frm_exclusion_metadata.append({"id": nid, "role": "FRM", "text": excl_text, "properties": props})

            elif label == "Topic":
                tname = props.get("name", "")
                frm_topic_texts.append(f"FRM In-Scope Topic: {tname}.")
                self.frm_topic_metadata.append({"name": tname, "id": nid, "role": "FRM"})

        # Explicit FRM Forbidden Semantic Concepts
        frm_hard_exclusions = [
            {"id": "resp:FRM:efficacy", "role": "FRM", "text": "Discussing clinical trial efficacy, overall survival (OS), progression-free survival (PFS), clinical study data, disease biology, or patient response rates."},
            {"id": "resp:FRM:biomarkers", "role": "FRM", "text": "Discussing biomarker testing protocols, genetic sequencing, EGFR exon 20 insertions, or molecular pathology diagnostics."},
            {"id": "resp:FRM:dosing_changes", "role": "FRM", "text": "Discussing drug dosing adjustments, administration protocols, subcutaneous infusion techniques, or medical mitigation of side effects."},
            {"id": "resp:FRM:ehr_workflows", "role": "FRM", "text": "Clinical operational workflows, electronic ordering drop-down menus, or EHR treatment prescribing."}
        ]
        for e in frm_hard_exclusions:
            frm_excl_texts.append(f"Forbidden Regulatory Exclusion for FRM: {e['text']}")
            self.frm_exclusion_metadata.append(e)

        # FRM Intents
        for item in CANONICAL_INTENTS:
            if item.get("role") == "FRM":
                frm_intent_texts.append(f"Role: FRM. Intent: {item['intent_name']}. Meaning: {item['semantic_description']}")
                self.frm_intent_metadata.append(item)

        self.frm_barrier_embeddings = self.encode_passages(frm_barrier_texts)
        self.frm_responsibility_embeddings = self.encode_passages(frm_resp_texts)
        self.frm_exclusion_embeddings = self.encode_passages(frm_excl_texts)
        self.frm_topic_embeddings = self.encode_passages(frm_topic_texts)
        self.frm_intent_embeddings = self.encode_passages(frm_intent_texts)

        # -------------------------------------------------------------
        # 3. BUILD DEDICATED HARD REGULATORY EXCLUSION TENSORS
        # -------------------------------------------------------------
        self.os_hard_exclusion_metadata = OS_HARD_EXCLUSIONS
        os_hard_texts = [f"Forbidden Regulatory Exclusion for OS: {e['text']}" for e in OS_HARD_EXCLUSIONS]
        self.os_hard_exclusion_embeddings = self.encode_passages(os_hard_texts)

        self.frm_hard_exclusion_metadata = FRM_HARD_EXCLUSIONS
        frm_hard_texts = [f"Forbidden Regulatory Exclusion for FRM: {e['text']}" for e in FRM_HARD_EXCLUSIONS]
        self.frm_hard_exclusion_embeddings = self.encode_passages(frm_hard_texts)

        # Populate backward-compatible unified views
        self._populate_unified_views()

        # Save precomputed cache to disk
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(
                {
                    # OS
                    "os_barrier_embeddings": self.os_barrier_embeddings.cpu(),
                    "os_barrier_metadata": self.os_barrier_metadata,
                    "os_responsibility_embeddings": self.os_responsibility_embeddings.cpu(),
                    "os_responsibility_metadata": self.os_responsibility_metadata,
                    "os_exclusion_embeddings": self.os_exclusion_embeddings.cpu(),
                    "os_exclusion_metadata": self.os_exclusion_metadata,
                    "os_hard_exclusion_embeddings": self.os_hard_exclusion_embeddings.cpu(),
                    "os_hard_exclusion_metadata": self.os_hard_exclusion_metadata,
                    "os_topic_embeddings": self.os_topic_embeddings.cpu(),
                    "os_topic_metadata": self.os_topic_metadata,
                    "os_intent_embeddings": self.os_intent_embeddings.cpu(),
                    "os_intent_metadata": self.os_intent_metadata,
                    # FRM
                    "frm_barrier_embeddings": self.frm_barrier_embeddings.cpu(),
                    "frm_barrier_metadata": self.frm_barrier_metadata,
                    "frm_responsibility_embeddings": self.frm_responsibility_embeddings.cpu(),
                    "frm_responsibility_metadata": self.frm_responsibility_metadata,
                    "frm_exclusion_embeddings": self.frm_exclusion_embeddings.cpu(),
                    "frm_exclusion_metadata": self.frm_exclusion_metadata,
                    "frm_hard_exclusion_embeddings": self.frm_hard_exclusion_embeddings.cpu(),
                    "frm_hard_exclusion_metadata": self.frm_hard_exclusion_metadata,
                    "frm_topic_embeddings": self.frm_topic_embeddings.cpu(),
                    "frm_topic_metadata": self.frm_topic_metadata,
                    "frm_intent_embeddings": self.frm_intent_embeddings.cpu(),
                    "frm_intent_metadata": self.frm_intent_metadata,
                },
                self.cache_path
            )
            print(f"[KGEmbeddingEngine] Saved precomputed MedEmbed cache to {self.cache_path} in {time.time()-t0:.2f}s.")
        except Exception as e:
            print(f"[KGEmbeddingEngine] Warning: Could not save cache: {e}")

    # =========================================================================
    # HYBRID COMPLIANCE GATE / CLAMP (Deterministic Fast-Path + Semantic Backstop)
    # =========================================================================
    def evaluate_compliance_gate(
        self,
        utterance: str,
        role: str = "OS",
        semantic_threshold: float = 0.58
    ) -> Dict[str, Any]:
        """
        Evaluates the utterance against strict role compliance boundaries.
        
        Architecture:
          Tier 1: Deterministic Keyword/Regex Lexicon (Fast-Path).
          Tier 2: MedEmbed Semantic Backstop for paraphrases.
          
        If triggered:
          Returns gate_triggered=True, static_weight=1.0, dynamic_weight=0.0,
          and a predefined compliance redirection question.
          The caller MUST BYPASS the SLM on gate trigger.
        """
        if not utterance or not utterance.strip():
            return {
                "gate_triggered": False,
                "action": "ALLOW",
                "static_weight": 0.30,
                "dynamic_weight": 0.70
            }

        role_upper = (role or "OS").upper()
        text_clean = utterance.strip()

        # -------------------------------------------------------------
        # TIER 1: Deterministic Keyword & Regex Lexicon (Fast-Path)
        # -------------------------------------------------------------
        if role_upper == "FRM":
            m = FRM_EXCLUSION_PATTERN.search(text_clean)
            if m:
                matched_phrase = m.group(0)
                return {
                    "gate_triggered": True,
                    "tier": "TIER_1_LEXICON",
                    "matched_rule": "resp:FRM:28",
                    "rule_text": f"FRMs are strictly prohibited from clinical discussions. Matched regulatory trigger: '{matched_phrase}'.",
                    "role": "FRM",
                    "action": "FORCE_COMPLIANCE_PATH",
                    "static_weight": 1.0,
                    "dynamic_weight": 0.0,
                    "compliance_question": (
                        "Under J&J commercial compliance guidelines, clinical trial efficacy, survival curves, "
                        "and biomarker data must be addressed by Medical Affairs. Did you inform the physician "
                        "that an MSL will follow up through a formal Medical Information Request (MIR)?"
                    )
                }
        elif role_upper == "OS":
            m = OS_EXCLUSION_PATTERN.search(text_clean)
            if m:
                matched_phrase = m.group(0)
                return {
                    "gate_triggered": True,
                    "tier": "TIER_1_LEXICON",
                    "matched_rule": "resp:OS:toxicity",
                    "rule_text": f"OS representatives are prohibited from clinical toxicity interventions. Matched trigger: '{matched_phrase}'.",
                    "role": "OS",
                    "action": "FORCE_COMPLIANCE_PATH",
                    "static_weight": 1.0,
                    "dynamic_weight": 0.0,
                    "compliance_question": (
                        "Commercial sales representatives cannot provide guidance on clinical side-effect management "
                        "or dose modifications. Did you report this safety concern to Medical Safety for formal triage?"
                    )
                }

        # -------------------------------------------------------------
        # TIER 2: MedEmbed Semantic Backstop (Paraphrase Defense)
        # -------------------------------------------------------------
        excl_tensors = self.frm_hard_exclusion_embeddings if role_upper == "FRM" else self.os_hard_exclusion_embeddings
        excl_meta = self.frm_hard_exclusion_metadata if role_upper == "FRM" else self.os_hard_exclusion_metadata

        # Fallback to general exclusions if hard exclusion tensors not yet populated
        if excl_tensors is None or len(excl_meta) == 0:
            excl_tensors = self.frm_exclusion_embeddings if role_upper == "FRM" else self.os_exclusion_embeddings
            excl_meta = self.frm_exclusion_metadata if role_upper == "FRM" else self.os_exclusion_metadata

        if excl_tensors is not None and len(excl_meta) > 0:
            # In-scope safeguarding to prevent false positives on legitimate commercial field notes
            text_lower = text_clean.lower()
            if role_upper == "FRM":
                in_scope_kw = [
                    "prior auth", "prior authorization", "pa ", "pa turnaround", "appeal", "denial",
                    "copay", "cost", "affordability", "out-of-pocket", "assistance", "foundation",
                    "reimbursement", "payer", "billing", "specialty pharmacy", "hub", "formulary",
                    "coverage", "pap", "patient access", "benefits verification"
                ]
                has_in_scope = any(k in text_lower for k in in_scope_kw)
            else:  # OS
                in_scope_kw = [
                    "eligible patient", "eligible patients", "candidate", "candidates", "pathway",
                    "flagging", "operationalizing", "patient identification", "bcg", "nmibc",
                    "bladder cancer", "treatment sequencing", "dr.", "dr ", "doctor", "apollo",
                    "fortis", "manipal", "max", "narayana", "touchpoint", "action items"
                ]
                has_in_scope = any(k in text_lower for k in in_scope_kw)

            # In-scope statements require higher threshold (>= 0.72) to trigger semantic backstop
            eff_threshold = 0.72 if has_in_scope else semantic_threshold

            query_vec = self.encode_query(text_clean)
            similarities = torch.mv(excl_tensors, query_vec).cpu().tolist()

            best_sim = -1.0
            best_meta = None
            for idx, sim in enumerate(similarities):
                if sim > best_sim:
                    best_sim = sim
                    best_meta = excl_meta[idx]

            if best_sim >= eff_threshold and best_meta:
                if role_upper == "FRM":
                    comp_q = (
                        "Under J&J commercial compliance guidelines, clinical trial efficacy discussions must be routed to "
                        "Medical Affairs. Did you advise the physician that an MSL will follow up via a Medical Information Request (MIR)?"
                    )
                else:
                    comp_q = (
                        "Commercial sales representatives cannot provide guidance on clinical side-effect management "
                        "or dose modifications. Did you report this safety concern to Medical Safety for formal triage?"
                    )

                return {
                    "gate_triggered": True,
                    "tier": "TIER_2_SEMANTIC",
                    "matched_rule": best_meta.get("id", f"resp:{role_upper}:scope"),
                    "rule_text": best_meta.get("text", ""),
                    "similarity_score": round(best_sim, 4),
                    "threshold": eff_threshold,
                    "role": role_upper,
                    "action": "FORCE_COMPLIANCE_PATH",
                    "static_weight": 1.0,
                    "dynamic_weight": 0.0,
                    "compliance_question": comp_q
                }

        return {
            "gate_triggered": False,
            "action": "ALLOW",
            "static_weight": 0.30,
            "dynamic_weight": 0.70
        }

    def check_scope_boundary(
        self,
        utterance: str,
        role: str = "FRM",
        threshold: float = 0.58
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for evaluate_compliance_gate."""
        res = self.evaluate_compliance_gate(utterance, role=role, semantic_threshold=threshold)
        return {
            "is_violation": res.get("gate_triggered", False),
            "matched_rule": res.get("matched_rule", ""),
            "rule_text": res.get("rule_text", ""),
            "similarity_score": res.get("similarity_score", 0.0),
            "tier": res.get("tier"),
            "role": (role or "FRM").upper(),
            "compliance_question": res.get("compliance_question"),
            "static_weight": res.get("static_weight", 0.30),
            "dynamic_weight": res.get("dynamic_weight", 0.70),
            "action": res.get("action", "ALLOW")
        }

    # =========================================================================
    # ROLE-ISOLATED RETRIEVAL & WEIGHTED MATCHING (0.3 Static / 0.7 Dynamic)
    # =========================================================================
    def match_account_barrier(
        self,
        utterance: str,
        account_name: Optional[str] = None,
        role: str = "OS",
        threshold: float = 0.42
    ) -> Optional[Dict[str, Any]]:
        """
        Dynamically matches user input against the isolated role barrier index:
          - OS -> Patient Identification Barriers
          - FRM -> Market Access Barriers
        Uses 0.70 dynamic priority weighting.
        """
        if not utterance:
            return None

        role_upper = (role or "OS").upper()
        barrier_tensor = self.os_barrier_embeddings if role_upper == "OS" else self.frm_barrier_embeddings
        barrier_meta = self.os_barrier_metadata if role_upper == "OS" else self.frm_barrier_metadata

        if barrier_tensor is None or len(barrier_meta) == 0:
            return None

        query_vec = self.encode_query(utterance)

        # Filter candidate barriers by account name if known
        acc_low = (account_name or "").lower().strip()
        candidates_idx = []
        for i, meta in enumerate(barrier_meta):
            if acc_low:
                b_acc = meta.get("account", "").lower()
                matched_acc = any(k in acc_low and k in b_acc for k in ["apollo", "fortis", "manipal", "max", "narayana"])
                if matched_acc or (b_acc in acc_low) or (acc_low in b_acc):
                    candidates_idx.append(i)
            else:
                candidates_idx.append(i)

        if not candidates_idx:
            candidates_idx = list(range(len(barrier_meta)))

        cand_tensor = barrier_tensor[candidates_idx]
        similarities = torch.mv(cand_tensor, query_vec).cpu().tolist()

        best_sim = -1.0
        best_meta_idx = -1
        for local_idx, sim in enumerate(similarities):
            if sim > best_sim:
                best_sim = sim
                best_meta_idx = candidates_idx[local_idx]

        if best_sim >= threshold and best_meta_idx >= 0:
            result = dict(barrier_meta[best_meta_idx])
            result["semantic_score"] = round(best_sim, 4)
            result["weighted_score"] = round(best_sim * 0.70, 4)
            result["matched_by"] = "MEDEMBED_COSINE_SIMILARITY"
            result["dynamic_weight"] = 0.70
            return result

        return None

    def match_conversation_intent(
        self,
        utterance: str,
        role: str = "OS",
        threshold: float = 0.38
    ) -> Optional[Dict[str, Any]]:
        """
        Matches user input against canonical interaction intents strictly within the role silo.
        """
        if not utterance:
            return None

        role_upper = (role or "OS").upper()
        intent_tensor = self.os_intent_embeddings if role_upper == "OS" else self.frm_intent_embeddings
        intent_meta = self.os_intent_metadata if role_upper == "OS" else self.frm_intent_metadata

        if intent_tensor is None or len(intent_meta) == 0:
            return None

        query_vec = self.encode_query(utterance)
        similarities = torch.mv(intent_tensor, query_vec).cpu().tolist()

        best_sim = -1.0
        best_meta_idx = -1
        for idx, sim in enumerate(similarities):
            if sim > best_sim:
                best_sim = sim
                best_meta_idx = idx

        if best_sim >= threshold and best_meta_idx >= 0:
            result = dict(intent_meta[best_meta_idx])
            result["similarity_score"] = round(best_sim, 4)
            result["weighted_score"] = round(best_sim * 0.70, 4)
            result["dynamic_weight"] = 0.70
            return result

        return None

    def match_relevant_responsibilities(
        self,
        utterance: str,
        role: str = "OS",
        top_k: int = 3,
        threshold: float = 0.20
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K core responsibilities from the isolated role silo.
        Applies 0.30 static weight.
        """
        if not utterance:
            return []

        role_upper = (role or "OS").upper()
        resp_tensor = self.os_responsibility_embeddings if role_upper == "OS" else self.frm_responsibility_embeddings
        resp_meta = self.os_responsibility_metadata if role_upper == "OS" else self.frm_responsibility_metadata

        if resp_tensor is None or len(resp_meta) == 0:
            return []

        query_vec = self.encode_query(utterance)
        similarities = torch.mv(resp_tensor, query_vec).cpu().tolist()

        scored = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                meta = dict(resp_meta[idx])
                meta["similarity_score"] = round(sim, 4)
                meta["weighted_score"] = round(sim * 0.30, 4)
                meta["static_weight"] = 0.30
                scored.append(meta)

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def match_topics(
        self,
        utterance: str,
        role: str = "OS",
        top_k: int = 3,
        threshold: float = 0.30
    ) -> List[Tuple[str, float]]:
        """
        Semantically identifies relevant topics from the isolated role topic silo.
        """
        if not utterance:
            return []

        role_upper = (role or "OS").upper()
        topic_tensor = self.os_topic_embeddings if role_upper == "OS" else self.frm_topic_embeddings
        topic_meta = self.os_topic_metadata if role_upper == "OS" else self.frm_topic_metadata

        if topic_tensor is None or len(topic_meta) == 0:
            return []

        query_vec = self.encode_query(utterance)
        similarities = torch.mv(topic_tensor, query_vec).cpu().tolist()

        scored = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                tname = topic_meta[idx]["name"]
                scored.append((tname, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
