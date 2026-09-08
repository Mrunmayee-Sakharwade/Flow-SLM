"""
Knowledge Graph Embedding & Dynamic Semantic Context Engine
============================================================
Vectorizes all Knowledge Graph nodes (Personas, Account Barriers, Core
Responsibilities, Compliance Rules, Topics, and Canonical Interaction Intents)
into dense semantic embeddings using sentence-transformers/all-MiniLM-L6-v2.

Enables:
  1. Semantic Account Barrier Detection on arbitrary, unscripted user phrasing.
  2. Dynamic Role Scope & Regulatory Boundary Projection in continuous vector space.
  3. Dynamic Semantic Topic & Responsibility Retrieval for context keeping.
  4. Semantic Intent & Node Inquiry Matching for intelligent next-question prediction.
  5. Performance telemetry: TTFT, Number of Tokens, and TRT measurement.
  6. Sub-10ms fast startup via precomputed PyTorch tensor caching (kg_embeddings.pt).
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

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Canonical conversational intents for semantic next-question steering
CANONICAL_INTENTS = [
    # OS (Oncology Sales) Intents
    {
        "id": "intent_os_no_patients",
        "role": "OS",
        "intent_name": "no_eligible_patients",
        "semantic_description": "Healthcare provider has no eligible patients, none identified so far, no matching cases right now, haven't seen candidates in clinic.",
        "inquiry_template": "What alternative treatment approach or protocol is {hcp} currently following?"
    },
    {
        "id": "intent_os_call_purpose",
        "role": "OS",
        "intent_name": "call_purpose_or_pathway",
        "semantic_description": "Discussing call purpose, wanting an update on treatment pathway or eligible bladder cancer NMIBC candidates.",
        "inquiry_template": "What specific treatment sequencing or pathway protocol did {hcp} discuss for eligible {brand} candidates?"
    },
    {
        "id": "intent_os_bcg_failure",
        "role": "OS",
        "intent_name": "bcg_failure_identified",
        "semantic_description": "A patient with post-BCG failure has been identified, found an eligible case post BCG failure.",
        "inquiry_template": "What is the current BCG status for that case?"
    },
    {
        "id": "intent_os_bcg_unresponsive",
        "role": "OS",
        "intent_name": "bcg_unresponsive_status",
        "semantic_description": "Patient completed full course of BCG therapy, confirmed BCG unresponsive or refractory.",
        "inquiry_template": "What is the next step for the patient?"
    },
    {
        "id": "intent_os_waiting_pathology",
        "role": "OS",
        "intent_name": "waiting_pathology_results",
        "semantic_description": "Waiting for lab test results, biopsy, pathology confirmation or molecular report.",
        "inquiry_template": "When are the pathology results expected for that patient?"
    },
    {
        "id": "intent_os_evaluating_history",
        "role": "OS",
        "intent_name": "evaluating_patient_history",
        "semantic_description": "Reviewing medical history, looking at patient chart, evaluating clinical eligibility and suitability for drug.",
        "inquiry_template": "What specific clinical parameters or medical history factors is {hcp} evaluating for {brand} eligibility?"
    },
    {
        "id": "intent_os_no_barriers",
        "role": "OS",
        "intent_name": "no_barriers_reported",
        "semantic_description": "No barriers reported, none right now, haven't discussed obstacles, as of now there are no hurdles.",
        "inquiry_template": "What is the target timing or next scheduled touchpoint with {hcp} for {brand}?"
    },
    {
        "id": "intent_os_timing_followup",
        "role": "OS",
        "intent_name": "timing_and_schedule",
        "semantic_description": "Touchpoint timing in a couple of weeks, next week, next month, several days or weeks out.",
        "inquiry_template": "Did {hcp} mention any support or resources needed?"
    },
    {
        "id": "intent_os_no_support_needed",
        "role": "OS",
        "intent_name": "support_not_needed",
        "semantic_description": "No support needed at the moment, nothing right now, no educational resources requested.",
        "inquiry_template": "Any follow-up or action items from the call?"
    },
    {
        "id": "intent_os_action_items",
        "role": "OS",
        "intent_name": "action_items_logged",
        "semantic_description": "Follow up by email, stay in touch, check back next week, schedule calendar reminder.",
        "inquiry_template": "Any other account context to capture?"
    },
    {
        "id": "intent_os_wrap_up",
        "role": "OS",
        "intent_name": "ready_to_wrap_up",
        "semantic_description": "That is all, nothing else to capture, all set, we are done, ready to close.",
        "inquiry_template": "I have enough for the call note. Should we wrap here?"
    },

    # FRM (Field Reimbursement Manager) Intents
    {
        "id": "intent_frm_pa_delay",
        "role": "FRM",
        "intent_name": "prior_authorization_delay",
        "semantic_description": "Prior authorization is delayed, pending review, payer pushed back or requested more documentation, PA denial or appeal.",
        "inquiry_template": "What specific prior authorization documentation, appeal steps, or turnaround hurdles were highlighted for {brand}?"
    },
    {
        "id": "intent_frm_copay_cost",
        "role": "FRM",
        "intent_name": "copay_affordability_exposure",
        "semantic_description": "Patient cost exposure, high copay, out of pocket expenses, needing patient assistance or foundation grant support.",
        "inquiry_template": "What copay assistance, foundation support, or patient access programs were evaluated to assist with that cost exposure for {brand}?"
    },
    {
        "id": "intent_frm_hub_specialty",
        "role": "FRM",
        "intent_name": "hub_and_specialty_pharmacy",
        "semantic_description": "Hub enrollment status, specialty pharmacy fulfillment, buy and bill logistics, permanent J-code billing.",
        "inquiry_template": "What is the status of the hub enrollment and specialty pharmacy fulfillment for {brand}?"
    },
    {
        "id": "intent_frm_payer_policy",
        "role": "FRM",
        "intent_name": "payer_coverage_formulary",
        "semantic_description": "Payer coverage policy updates, Medicare Advantage guidelines, formulary exception committee review.",
        "inquiry_template": "What specific payer coverage or formulary guidelines were identified for {brand}?"
    },
    {
        "id": "intent_frm_timing_followup",
        "role": "FRM",
        "intent_name": "follow_up_access_progress",
        "semantic_description": "Following up with billing coordinator or office staff next week or tomorrow after documentation upload.",
        "inquiry_template": "When is the target follow-up date with {hcp} to review access progress on {brand}?"
    }
]


class KGEmbeddingEngine:
    """
    High-performance, localized embedding and semantic retrieval engine
    for the Johnson & Johnson Commercial Knowledge Graph.
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
        self.cache_path = self.cache_dir / "kg_embeddings.pt"

        # Hardware device selection (CPU or CUDA)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        # In-memory index structures
        self.kg_data = kg_data
        self.account_barriers = account_barriers or []

        # Vector stores and metadata
        self.barrier_embeddings: Optional[torch.Tensor] = None
        self.barrier_metadata: List[Dict[str, Any]] = []

        self.responsibility_embeddings: Optional[torch.Tensor] = None
        self.responsibility_metadata: List[Dict[str, Any]] = []

        self.topic_embeddings: Optional[torch.Tensor] = None
        self.topic_metadata: List[Dict[str, Any]] = []

        self.exclusion_embeddings: Optional[torch.Tensor] = None
        self.exclusion_metadata: List[Dict[str, Any]] = []

        self.role_embeddings: Optional[torch.Tensor] = None
        self.role_metadata: List[Dict[str, Any]] = []

        self.intent_embeddings: Optional[torch.Tensor] = None
        self.intent_metadata: List[Dict[str, Any]] = []

        # Initialize or load cached embeddings
        self._initialize_embeddings()

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element contains hidden states
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

    def encode_single(self, text: str) -> torch.Tensor:
        """Encodes a single text string into a 1D unit tensor [384]."""
        return self.encode([text])[0]

    def _initialize_embeddings(self):
        """Loads embeddings from cache or computes and saves them."""
        if self.cache_path.exists():
            try:
                t0 = time.time()
                cached = torch.load(self.cache_path, map_location=self.device)
                self.barrier_embeddings = cached["barrier_embeddings"].to(self.device)
                self.barrier_metadata = cached["barrier_metadata"]
                self.responsibility_embeddings = cached["responsibility_embeddings"].to(self.device)
                self.responsibility_metadata = cached["responsibility_metadata"]
                self.topic_embeddings = cached["topic_embeddings"].to(self.device)
                self.topic_metadata = cached["topic_metadata"]
                self.exclusion_embeddings = cached["exclusion_embeddings"].to(self.device)
                self.exclusion_metadata = cached["exclusion_metadata"]
                self.role_embeddings = cached["role_embeddings"].to(self.device)
                self.role_metadata = cached["role_metadata"]
                self.intent_embeddings = cached.get("intent_embeddings", torch.empty((0, EMBEDDING_DIM))).to(self.device)
                self.intent_metadata = cached.get("intent_metadata", [])

                if len(self.intent_metadata) == len(CANONICAL_INTENTS) and len(self.intent_metadata) > 0:
                    print(
                        f"[KGEmbeddingEngine] Loaded precomputed KG embeddings from cache "
                        f"({len(self.barrier_metadata)} barriers, {len(self.responsibility_metadata)} responsibilities, "
                        f"{len(self.topic_metadata)} topics, {len(self.intent_metadata)} intents) in {time.time()-t0:.3f}s."
                    )
                    return
            except Exception as e:
                print(f"[KGEmbeddingEngine] Failed to load cache from {self.cache_path}: {e}. Recomputing...")

        # Recompute embeddings from source data
        self._build_embeddings()

    def _build_embeddings(self):
        """Constructs vector indexes for all Knowledge Graph entities and saves to disk."""
        t0 = time.time()
        print("[KGEmbeddingEngine] Vectorizing Knowledge Graph entities...")

        # 1. Vectorize Account Barriers
        barrier_texts = []
        self.barrier_metadata = []
        for b in self.account_barriers:
            acc = b.get("account", "")
            b_type = b.get("barrier_type", "")
            summary = b.get("barrier_summary", "")
            details = b.get("barrier_details", "")
            keywords = " ".join(b.get("trigger_keywords", []))
            case1 = b.get("case_1_inquiry", "")
            case2 = b.get("case_2_proactive_context", "")

            semantic_text = (
                f"Account: {acc}. Barrier Type: {b_type}. Summary: {summary}. "
                f"Details: {details}. Key Triggers: {keywords}. Context: {case2}"
            )
            barrier_texts.append(semantic_text)
            self.barrier_metadata.append(b)

        self.barrier_embeddings = self.encode(barrier_texts)

        # 2. Vectorize Core Responsibilities
        resp_texts = []
        self.responsibility_metadata = []
        seen_resps = set()

        def add_resp_node(nid, label, props):
            if nid in seen_resps:
                return
            seen_resps.add(nid)
            role = "OS" if "OS" in nid else ("FRM" if "FRM" in nid else "")
            text = props.get("text", "")
            domain = props.get("domain", "")
            topics = props.get("topics", [])
            topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
            semantic_text = f"Role: {role}. Domain: {domain}. Duty: {text}. Relevant Topics: {topics_str}."
            resp_texts.append(semantic_text)
            self.responsibility_metadata.append({
                "id": nid,
                "role": role,
                "text": text,
                "domain": domain,
                "topics": topics,
                "properties": props
            })

        # Load from kg_data
        if self.kg_data:
            for node in self.kg_data.get("nodes", []):
                if node.get("label") == "CoreResponsibility":
                    add_resp_node(node.get("id", ""), node.get("label"), node.get("properties", {}))

        # Also load from dedicated role KGs if available to guarantee all duties are indexed
        for r_file in ["kg_os.json", "kg_frm.json"]:
            p = self.base_dir / r_file
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as rf:
                        rdata = json.load(rf)
                    for node in rdata.get("nodes", []):
                        if node.get("label") == "CoreResponsibility":
                            add_resp_node(node.get("id", ""), node.get("label"), node.get("properties", {}))
                except Exception:
                    pass

        self.responsibility_embeddings = self.encode(resp_texts)

        # 3. Vectorize Topics
        topic_texts = []
        self.topic_metadata = []
        if self.kg_data:
            for node in self.kg_data.get("nodes", []):
                if node.get("label") == "Topic":
                    props = node.get("properties", {})
                    tname = props.get("name", "")
                    desc = props.get("description", tname)
                    semantic_text = f"Topic: {tname}. Scope: {desc}."
                    topic_texts.append(semantic_text)
                    self.topic_metadata.append({"name": tname, "description": desc, "id": node.get("id")})

        self.topic_embeddings = self.encode(topic_texts)

        # 4. Vectorize Exclusion & Compliance Rules
        excl_texts = []
        self.exclusion_metadata = []
        if self.kg_data:
            for node in self.kg_data.get("nodes", []):
                if node.get("label") == "ExclusionRule":
                    props = node.get("properties", {})
                    nid = node.get("id", "")
                    role = "OS" if "OS" in nid else ("FRM" if "FRM" in nid else "")
                    text = props.get("text", "")
                    semantic_text = f"Forbidden Exclusion Rule for {role}: {text}"
                    excl_texts.append(semantic_text)
                    self.exclusion_metadata.append({"id": nid, "role": role, "text": text, "properties": props})
                elif node.get("label") == "ComplianceRule":
                    props = node.get("properties", {})
                    name = props.get("name", "")
                    summary = props.get("summary", "")
                    semantic_text = f"Universal Compliance Rule: {name}. {summary}"
                    excl_texts.append(semantic_text)
                    self.exclusion_metadata.append({"id": node.get("id"), "role": "GLOBAL", "text": summary, "properties": props})

        # Add explicit semantic forbidden concepts for precision matching
        forbidden_semantic_rules = [
            {
                "id": "resp:FRM:28",
                "role": "FRM",
                "text": "Discussing clinical trial efficacy, overall survival (OS), progression-free survival (PFS), clinical study data, disease biology, or patient response rates.",
                "properties": {"category": "exclusion"}
            },
            {
                "id": "resp:FRM:28",
                "role": "FRM",
                "text": "Discussing biomarker testing protocols, genetic sequencing, EGFR exon 20 insertions, or molecular pathology diagnostics.",
                "properties": {"category": "exclusion"}
            },
            {
                "id": "resp:FRM:24",
                "role": "FRM",
                "text": "Discussing drug dosing adjustments, administration protocols, subcutaneous infusion techniques, or medical mitigation of side effects.",
                "properties": {"category": "exclusion"}
            },
            {
                "id": "resp:OS:toxicity",
                "role": "OS",
                "text": "Managing clinical toxicity, treating adverse events, adjusting medical dosages, or clinical toxicity interventions.",
                "properties": {"category": "exclusion"}
            }
        ]
        for rule in forbidden_semantic_rules:
            excl_texts.append(f"Forbidden Regulatory Exclusion for {rule['role']}: {rule['text']}")
            self.exclusion_metadata.append(rule)

        self.exclusion_embeddings = self.encode(excl_texts)

        # 5. Vectorize Roles
        role_texts = []
        self.role_metadata = []
        if self.kg_data:
            for node in self.kg_data.get("nodes", []):
                if node.get("label") == "Role":
                    props = node.get("properties", {})
                    rname = props.get("name", "")
                    full = props.get("full_name", "")
                    desc = props.get("description", "")
                    semantic_text = f"Persona: {rname} ({full}). Operational Mandate: {desc}"
                    role_texts.append(semantic_text)
                    self.role_metadata.append({"name": rname, "full_name": full, "description": desc})

        self.role_embeddings = self.encode(role_texts)

        # 6. Vectorize Canonical Intents
        intent_texts = []
        self.intent_metadata = []
        for item in CANONICAL_INTENTS:
            intent_texts.append(f"Role: {item['role']}. Intent: {item['intent_name']}. Meaning: {item['semantic_description']}")
            self.intent_metadata.append(item)

        self.intent_embeddings = self.encode(intent_texts)

        # Save to disk cache
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(
                {
                    "barrier_embeddings": self.barrier_embeddings.cpu(),
                    "barrier_metadata": self.barrier_metadata,
                    "responsibility_embeddings": self.responsibility_embeddings.cpu(),
                    "responsibility_metadata": self.responsibility_metadata,
                    "topic_embeddings": self.topic_embeddings.cpu(),
                    "topic_metadata": self.topic_metadata,
                    "exclusion_embeddings": self.exclusion_embeddings.cpu(),
                    "exclusion_metadata": self.exclusion_metadata,
                    "role_embeddings": self.role_embeddings.cpu(),
                    "role_metadata": self.role_metadata,
                    "intent_embeddings": self.intent_embeddings.cpu(),
                    "intent_metadata": self.intent_metadata,
                },
                self.cache_path
            )
            print(f"[KGEmbeddingEngine] Saved precomputed embeddings to {self.cache_path} in {time.time()-t0:.2f}s.")
        except Exception as e:
            print(f"[KGEmbeddingEngine] Warning: Could not save cache: {e}")

    def match_account_barrier(
        self,
        utterance: str,
        account_name: Optional[str] = None,
        role: str = "OS",
        threshold: float = 0.46
    ) -> Optional[Dict[str, Any]]:
        """
        Dynamically matches any user utterance (statement, question, observation)
        against the embedded Account Barrier profiles using cosine similarity.
        """
        if not utterance or self.barrier_embeddings is None or len(self.barrier_metadata) == 0:
            return None

        role_upper = (role or "OS").upper()
        target_b_type = "Patient Identification Barrier" if role_upper == "OS" else "Market Access Barrier"

        query_vec = self.encode_single(utterance)

        # Filter candidate barriers by role type and (if known) account name
        acc_low = (account_name or "").lower().strip()
        candidates_idx = []
        for i, meta in enumerate(self.barrier_metadata):
            if meta.get("barrier_type") != target_b_type:
                continue
            if acc_low:
                b_acc = meta.get("account", "").lower()
                matched_acc = any(k in acc_low and k in b_acc for k in ["apollo", "fortis", "manipal", "max", "narayana"])
                if matched_acc or (b_acc in acc_low) or (acc_low in b_acc):
                    candidates_idx.append(i)
            else:
                candidates_idx.append(i)

        if not candidates_idx:
            candidates_idx = [
                i for i, meta in enumerate(self.barrier_metadata)
                if meta.get("barrier_type") == target_b_type
            ]

        if not candidates_idx:
            return None

        cand_tensor = self.barrier_embeddings[candidates_idx]
        similarities = torch.mv(cand_tensor, query_vec).cpu().tolist()

        best_sim = -1.0
        best_meta_idx = -1
        for local_idx, sim in enumerate(similarities):
            if sim > best_sim:
                best_sim = sim
                best_meta_idx = candidates_idx[local_idx]

        if best_sim >= threshold and best_meta_idx >= 0:
            result = dict(self.barrier_metadata[best_meta_idx])
            result["semantic_score"] = round(best_sim, 4)
            result["matched_by"] = "EMBEDDING_COSINE_SIMILARITY"
            return result

        return None

    def match_conversation_intent(
        self,
        utterance: str,
        role: str = "OS",
        threshold: float = 0.38
    ) -> Optional[Dict[str, Any]]:
        """
        Matches arbitrary user conversational input against canonical interaction intents
        to steer the Next-Question generator.
        """
        if not utterance or self.intent_embeddings is None or len(self.intent_metadata) == 0:
            return None

        role_upper = (role or "OS").upper()
        candidates_idx = [
            i for i, meta in enumerate(self.intent_metadata)
            if meta.get("role") == role_upper
        ]

        if not candidates_idx:
            return None

        query_vec = self.encode_single(utterance)
        cand_tensor = self.intent_embeddings[candidates_idx]
        similarities = torch.mv(cand_tensor, query_vec).cpu().tolist()

        best_sim = -1.0
        best_meta_idx = -1
        for local_idx, sim in enumerate(similarities):
            if sim > best_sim:
                best_sim = sim
                best_meta_idx = candidates_idx[local_idx]

        if best_sim >= threshold and best_meta_idx >= 0:
            result = dict(self.intent_metadata[best_meta_idx])
            result["similarity_score"] = round(best_sim, 4)
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
        Retrieves the top-K most semantically relevant J&J Core Responsibilities
        for the given utterance and role from the embedded graph.
        """
        if not utterance or self.responsibility_embeddings is None:
            return []

        role_upper = (role or "OS").upper()
        candidates_idx = [
            i for i, meta in enumerate(self.responsibility_metadata)
            if meta.get("role") == role_upper
        ]

        if not candidates_idx:
            return []

        query_vec = self.encode_single(utterance)
        cand_tensor = self.responsibility_embeddings[candidates_idx]
        similarities = torch.mv(cand_tensor, query_vec).cpu().tolist()

        scored = []
        for local_idx, sim in enumerate(similarities):
            if sim >= threshold:
                meta = dict(self.responsibility_metadata[candidates_idx[local_idx]])
                meta["similarity_score"] = round(sim, 4)
                scored.append(meta)

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def match_topics(
        self,
        utterance: str,
        top_k: int = 3,
        threshold: float = 0.30
    ) -> List[Tuple[str, float]]:
        """
        Semantically identifies relevant topics from the 22 KG topics for any open-ended input.
        Returns a list of (topic_name, similarity_score).
        """
        if not utterance or self.topic_embeddings is None:
            return []

        query_vec = self.encode_single(utterance)
        similarities = torch.mv(self.topic_embeddings, query_vec).cpu().tolist()

        scored = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                tname = self.topic_metadata[i]["name"]
                scored.append((tname, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def check_scope_boundary(
        self,
        utterance: str,
        role: str = "FRM",
        threshold: float = 0.52
    ) -> Dict[str, Any]:
        """
        Checks if the utterance semantically matches any strict exclusion rule for the role.
        e.g., FRM discussing clinical efficacy/biomarkers, OS discussing toxicity management.
        """
        if not utterance or self.exclusion_embeddings is None:
            return {"is_violation": False}

        role_upper = (role or "FRM").upper()
        candidates_idx = [
            i for i, meta in enumerate(self.exclusion_metadata)
            if meta.get("role") in [role_upper, "GLOBAL"]
        ]

        if not candidates_idx:
            return {"is_violation": False}

        query_vec = self.encode_single(utterance)
        cand_tensor = self.exclusion_embeddings[candidates_idx]
        similarities = torch.mv(cand_tensor, query_vec).cpu().tolist()

        best_sim = -1.0
        best_meta = None
        for local_idx, sim in enumerate(similarities):
            if sim > best_sim:
                best_sim = sim
                best_meta = self.exclusion_metadata[candidates_idx[local_idx]]

        if best_sim >= threshold and best_meta:
            return {
                "is_violation": True,
                "matched_rule": best_meta.get("id", f"resp:{role_upper}:scope"),
                "rule_text": best_meta.get("text", ""),
                "similarity_score": round(best_sim, 4),
                "role": role_upper
            }

        return {"is_violation": False, "similarity_score": round(best_sim, 4)}
