"""
Knowledge Graph Context Retriever Engine (GraphRAG for SLM)
============================================================
Extracts, structures, and serializes rich context from the Persona Solid Cancer
Knowledge Graph (Neo4j ontology and scoped JSON) to ground the Small Language Model (SLM).

Provides:
  1. Dynamic Role Scope & Operational Boundaries (In-Scope vs. Strict Exclusion Rules)
  2. Active J&J Core Responsibilities matching current state & conversation topics
  3. Dynamic Topic Traversal (Covered Topics vs. Unaddressed Mandatory Topics)
  4. Intelligent Brand Detection (INLEXZO, RYBREVANT + LAZCLUZE, RYBREVANT)
  5. Anti-Repetition Guard: Prevents asking for topics/information already provided
  6. Cross-Functional Collaboration & Handoff Detection (COLLABORATES_WITH)
  7. Universal Compliance Directives (PHI, MIR, Toxicity Control)
  8. Flat Prompt Format (infer.py-compatible) for merged Llama-3.2-1B inference
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set
from engine.kg_embedding_engine import KGEmbeddingEngine

# ---------------------------------------------------------------------------
# Flat-Format System Prompt (aligned with infer.py / merged Llama-3.2-1B)
# ---------------------------------------------------------------------------
FLAT_SYSTEM_PROMPT = (
    "Given the conversation history and Knowledge Graph context, generate the single most "
    "appropriate next question for the AI to ask the field representative.\n\n"
    "Requirements:\n"
    "- Ask exactly one question.\n"
    "- The question must follow naturally from the latest user response.\n"
    "- The question must focus on the target topic and adhere to the active J&J Core Responsibility.\n"
    "- Use only information available in the conversation history.\n"
    "- Do not repeat questions that have already been asked or ask for information already provided.\n"
    "- Do not answer the user.\n"
    "- Do not explain, summarize, advise, recommend, or suggest anything.\n"
    "- NEVER say 'I can't capture' or refuse to record call details.\n"
    "- NEVER generate compliance warnings or refusal messages.\n\n"
    "Output only the next question."
)


def detect_brand_from_text(text: str) -> Optional[str]:
    """
    Intelligently auto-detects J&J oncology brands from conversation text or clinical keywords.
    - INLEXZO: NMIBC, bladder cancer, BCG-unresponsive, post-BCG failure, TAR-200, urology, insertion, instillation.
    - RYBREVANT + LAZCLUZE: 1L EGFR NSCLC, MARIPOSA, combination regimen, oral lazcluze.
    - RYBREVANT: Exon 20 insertion, EGFR NSCLC, infusion, subcutaneous, amivantamab.
    """
    text_lower = text.lower()
    
    # 1. Direct Brand Mentions
    if "inlexzo" in text_lower:
        return "INLEXZO"
    if "lazcluze" in text_lower and "rybrevant" in text_lower:
        return "RYBREVANT + LAZCLUZE"
    if "rybrevant" in text_lower:
        return "RYBREVANT"
    if "lazcluze" in text_lower:
        return "LAZCLUZE"
        
    # 2. Disease State & Indication Clues
    if any(k in text_lower for k in ["nmibc", "bladder", "baldder", "bcg", "bcg failure", "post bcg", "bcg-unresponsive", "urology", "urologist", "tar-200", "instillation", "insertion"]):
        return "INLEXZO"
    if any(k in text_lower for k in ["mariposa", "first-line egfr", "1l egfr", "tagrisso"]):
        return "RYBREVANT + LAZCLUZE"
    if any(k in text_lower for k in ["exon 20", "exon20", "amivantamab", "infusion reaction", "irr", "rash prophylaxis"]):
        return "RYBREVANT"
        
    return None


CANONICAL_ACCOUNTS = [
    ("atlantic", "Atlantic Urology Associates"),
    ("capital", "Capital Bladder Cancer Center"),
    ("central ohio", "Central Ohio Urology"),
    ("northside", "Northside Urology Group"),
    ("regional", "Regional Urology Institute"),
    ("summit", "Summit Urologic Oncology"),
    ("temple", "Temple Urology Clinic"),
    ("valley", "Valley Urology Specialists"),
]


class KGContextRetriever:
    def __init__(self, kg_json_path: str = "kg_os.json", init_embedder: bool = True):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Role-specific Knowledge Graph files
        self.role_kg_files = {
            "OS": os.path.join(base_dir, "kg_os.json")
        }
        self.role_kg_data = {}
        for r, p in self.role_kg_files.items():
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as rf:
                        self.role_kg_data[r] = json.load(rf)
                except Exception as e:
                    print(f"[Warning] Could not load {p}: {e}")

        if not os.path.isabs(kg_json_path):
            candidate = os.path.join(base_dir, kg_json_path)
            if os.path.exists(candidate):
                kg_json_path = candidate
                
        with open(kg_json_path, "r", encoding="utf-8") as f:
            self.kg_data = json.load(f)
            
        self.roles = {}
        self.domains = {}
        self.topics = {}
        self.accounts = {}
        self.in_scope_topics = {"OS": set(), "FRM": set()}
        self.out_of_scope_topics = {"OS": set(), "FRM": set()}
        self.core_responsibilities = {"OS": [], "FRM": []}
        self.exclusion_rules = {"OS": [], "FRM": []}
        self.compliance_rules = []
        self.collaborations = {"OS": [], "FRM": []}
        self.last_barrier_case: Optional[str] = None
        
        # Topic-to-Responsibilities index
        self.topic_to_resp = defaultdict(lambda: {"OS": [], "FRM": []})
        
        # Standard sequential call topic roadmap per role
        self.role_topic_roadmap = {
            "FRM": [
                "account identification",
                "patient access",
                "payer coverage",
                "prior authorization",
                "reimbursement",
                "affordability patient support",
                "cross functional collaboration",
                "territory engagement planning"
            ],
            "OS": [
                "account identification",
                "efficacy safety product info",
                "treatment sequencing",
                "cross functional collaboration",
                "territory engagement planning"
            ]
        }
        
        # Load Account Barrier Intelligence from dedicated KGs or fallback
        self.account_barriers = []
        for r, rdata in self.role_kg_data.items():
            for node in rdata.get("nodes", []):
                if node.get("label") == "AccountBarrier":
                    self.account_barriers.append(node.get("properties", {}))
                    
        if not self.account_barriers:
            barriers_path = os.path.join(base_dir, "account_barriers.json")
            if os.path.exists(barriers_path):
                try:
                    with open(barriers_path, "r", encoding="utf-8") as bf:
                        self.account_barriers = json.load(bf)
                except Exception as e:
                    print(f"[Warning] Could not load account_barriers.json: {e}")

        self._index_graph()

        # Dynamic Knowledge Graph Embedding Engine (optional for batch dataset generation)
        if init_embedder:
            try:
                self.embedder = KGEmbeddingEngine(kg_data=self.kg_data, account_barriers=self.account_barriers)
            except Exception as e:
                print(f"[Warning] Could not initialize KGEmbeddingEngine: {e}")
                self.embedder = None
        else:
            self.embedder = None

    def get_account_barrier(self, account_name: Optional[str], role: str = "OS") -> Optional[Dict[str, Any]]:
        """
        Retrieves the account barrier profile for the given account from the OS Knowledge Graph.
        Grounded in the 8 target accounts from the INLEXZO field transcripts.
        """
        if not account_name or not self.account_barriers:
            return None
        
        acc_low = account_name.lower().strip()
        
        # 1. Exact or substring match
        for b in self.account_barriers:
            b_acc = b.get("account", "").lower()
            if b_acc in acc_low or acc_low in b_acc:
                return b

        # 2. Canonical token match
        for k, canonical in CANONICAL_ACCOUNTS:
            if k in acc_low:
                for b in self.account_barriers:
                    if canonical.lower() == b.get("account", "").lower():
                        return b
        return None

    def detect_barrier_mention(self, text: str, account_name: Optional[str], role: str = "OS") -> Optional[Dict[str, Any]]:
        """
        Detects if the rep/HCP explicitly mentions the account barrier in text (Case 1).
        Matches against trigger keywords, key phrases from barrier details, or explicit barrier mentions.
        """
        if not text:
            return None
        b = self.get_account_barrier(account_name, role)
        if not b:
            text_low = text.lower()
            for k, canonical in CANONICAL_ACCOUNTS:
                if k in text_low:
                    b = self.get_account_barrier(canonical, role)
                    break
        if not b:
            return None

        text_lower = text.lower().strip()
        # If user explicitly states there are NO barriers, or that barriers were not discussed, do NOT treat as barrier mention!
        negation_cues = [
            "no barrier", "no barriers", "none as of now", "none right now",
            "no delay", "no delays", "not delayed", "no issue", "no issues", 
            "no concern", "no concerns", "didnt come up", "didn't come up", 
            "did not come up", "haven't discussed", "havent discussed", "not discussed"
        ]
        if any(neg in text_lower for neg in negation_cues):
            return None

        trigger_keywords = b.get("trigger_keywords", [])
        for kw in trigger_keywords:
            if kw.lower() in text_lower:
                return b
        
        # Check semantic phrases from barrier details
        details_lower = b.get("barrier_details", "").lower()
        role_upper = (role or "OS").upper()
        if "prior authorization" in details_lower and any(w in text_lower for w in ["pa ", "prior auth", "prior authorization", "pa holdup", "pa is still"]):
            return b
        if ("formulary" in details_lower or "p&t" in details_lower) and any(w in text_lower for w in ["formulary", "p&t", "committee", "committee review", "exception"]):
            return b
        if "coverage" in details_lower and any(w in text_lower for w in ["coverage", "benefits investigation", "benefits verification", "patient responsibility", "rechecking"]):
            return b
        if "deductible" in details_lower and any(w in text_lower for w in ["deductible", "affordability", "payment timing", "payment plan"]):
            return b
        if ("prep" in details_lower or "anatomical" in details_lower) and any(w in text_lower for w in ["anatomical", "prep session", "nurse prep", "staff prep", "anatomical model"]):
            return b

        # Semantic Embedding Match Fallback for arbitrary/unscripted phrasing
        # Ensure utterance actually describes friction/hurdles/challenges, not just meeting intro / account capture
        friction_cues = [
            "barrier", "friction", "delay", "delays", "delayed", "challenge", "challenges",
            "trouble", "issue", "issues", "problem", "problems", "hurdle", "hurdles",
            "difficulty", "difficult", "hard", "struggling", "struggle", "uncertain",
            "uncertainty", "hesitation", "concern", "concerns", "exposure", "rework",
            "restricted", "exception", "denial", "denials", "appeals", "inconsistent",
            "late", "slowing", "slow", "timing", "pending", "operationaliz", "eligible"
        ]
        has_friction = any(fc in text_lower for fc in friction_cues)
        if getattr(self, "embedder", None) and has_friction and len(text.split()) >= 4:
            sem_match = self.embedder.match_account_barrier(text, account_name=account_name, role=role_upper, threshold=0.46)
            if sem_match:
                return sem_match

        return None

    def generate_live_chat_subgraph(
        self,
        role: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        slots: Optional[Dict[str, Any]] = None,
        candidate_answer: str = "",
        detected_entities: Optional[Dict[str, Any]] = None,
        detected_topics: Optional[List[str]] = None,
        barrier_case: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dynamically constructs the live Entity-Relationship Knowledge Subgraph
        based on the current live chat context and extracted entities.
        Produces:
          - nodes: List[Dict] with id, label, type, name, status, and properties
          - edges: List[Dict] with source, target, label, status
          - mermaid: Formatted Mermaid diagram using dynamic Entity-Relationship syntax
        """
        role_upper = (role or "OS").upper()
        curr_slots = slots or {}
        ent = detected_entities or {}
        hist = conversation_history or []
        hist_text = " ".join([h.get("text", "") for h in hist]).lower() + " " + candidate_answer.lower()

        # 1. Resolve Active Entities
        rep_name = curr_slots.get("user_name") or f"{role_upper} Representative"
        resolved_account = curr_slots.get("account_name")
        if not resolved_account and ent.get("accounts"):
            resolved_account = ent["accounts"][0]
        if not resolved_account:
            for k, canonical in CANONICAL_ACCOUNTS:
                if k in hist_text:
                    resolved_account = canonical
                    break

        hcp_name = curr_slots.get("hcp_name") or (ent.get("hcps", [None])[0] if ent.get("hcps") else None)
        if not hcp_name:
            m = re.search(r'\b(Dr\.?\s+[A-Za-z]+)\b', hist_text, re.IGNORECASE)
            if m:
                hcp_name = m.group(1).title()

        brand_name = curr_slots.get("brand") or (ent.get("brands", [None])[0] if ent.get("brands") else None)
        if not brand_name:
            if any(w in hist_text for w in ["nmibc", "bladder", "bcg", "tar-200"]):
                brand_name = "INLEXZO"
            elif any(w in hist_text for w in ["nsclc", "lung", "egfr", "exon 20", "lazcluze"]):
                brand_name = "RYBREVANT"
            else:
                brand_name = "INLEXZO" if role_upper == "OS" else "RYBREVANT"

        acc_barrier = self.get_account_barrier(resolved_account, role_upper) if resolved_account else None

        # 2. Build Subgraph Nodes
        nodes = []
        edges = []

        # Node: Rep
        nodes.append({
            "id": "node_rep",
            "label": "Representative",
            "name": rep_name,
            "type": "rep",
            "status": "active"
        })

        # Node: Commercial Role
        nodes.append({
            "id": f"node_role_{role_upper.lower()}",
            "label": "Role",
            "name": f"{role_upper} Persona",
            "type": "role",
            "status": "active"
        })
        edges.append({
            "source": "node_rep",
            "target": f"node_role_{role_upper.lower()}",
            "label": "ASSIGNED_TO",
            "status": "active"
        })

        # Node: Account (if known)
        if resolved_account:
            nodes.append({
                "id": "node_account",
                "label": "Account",
                "name": resolved_account,
                "type": "account",
                "status": "covered" if hcp_name else "active"
            })
            edges.append({
                "source": "node_rep",
                "target": "node_account",
                "label": "VISITED",
                "status": "covered" if hcp_name else "active"
            })

        # Node: HCP (if known)
        if hcp_name:
            nodes.append({
                "id": "node_hcp",
                "label": "HCP",
                "name": hcp_name,
                "type": "hcp",
                "status": "active"
            })
            edges.append({
                "source": "node_rep",
                "target": "node_hcp",
                "label": "ENGAGED_WITH",
                "status": "active"
            })
            if resolved_account:
                edges.append({
                    "source": "node_hcp",
                    "target": "node_account",
                    "label": "AFFILIATED_WITH",
                    "status": "active"
                })

        # Node: Brand (if known)
        if brand_name:
            nodes.append({
                "id": "node_brand",
                "label": "Brand",
                "name": brand_name,
                "type": "brand",
                "status": "covered" if any(w in hist_text for w in ["eligible", "prior auth", "efficacy", "patient"]) else "active"
            })
            if hcp_name:
                edges.append({
                    "source": "node_hcp",
                    "target": "node_brand",
                    "label": "DISCUSSED",
                    "status": "active"
                })
            else:
                edges.append({
                    "source": "node_rep",
                    "target": "node_brand",
                    "label": "REPRESENTS",
                    "status": "active"
                })

        # Node: Patient Cohort / Indication (if clinical terms mentioned)
        has_cohort = any(w in hist_text for w in ["nmibc", "bladder", "bcg", "post bcg", "unresponsive", "nsclc", "lung", "exon 20"])
        if has_cohort:
            cohort_name = "NMIBC (BCG-Unresponsive)" if any(w in hist_text for w in ["nmibc", "bladder", "bcg"]) else "NSCLC (EGFR Exon 20)"
            nodes.append({
                "id": "node_cohort",
                "label": "Indication",
                "name": cohort_name,
                "type": "cohort",
                "status": "active"
            })
            edges.append({
                "source": "node_brand",
                "target": "node_cohort",
                "label": "INDICATED_FOR",
                "status": "active"
            })
            if hcp_name:
                edges.append({
                    "source": "node_hcp",
                    "target": "node_cohort",
                    "label": "IDENTIFIED_CASE",
                    "status": "active"
                })

        # Node: Access / PA Topic (for FRM or if PA mentioned)
        has_access = any(w in hist_text for w in ["prior auth", "pa", "payer", "coverage", "copay", "reimbursement"])
        if has_access or role_upper == "FRM":
            nodes.append({
                "id": "node_access",
                "label": "AccessTopic",
                "name": "Prior Authorization & Payer Criteria",
                "type": "topic",
                "status": "active"
            })
            edges.append({
                "source": "node_brand",
                "target": "node_access",
                "label": "GOVERNED_BY",
                "status": "active"
            })

        # Node: Account Barrier & Request
        if acc_barrier and resolved_account:
            effective_case = barrier_case or self.last_barrier_case
            is_case_1 = effective_case == "CASE_1_USER_INITIATED"
            is_case_2 = effective_case == "CASE_2_PROACTIVE_FOLLOWUP"
            barrier_status = "case_1_active" if is_case_1 else ("case_2_probed" if is_case_2 else "pending")
            
            nodes.append({
                "id": "node_barrier",
                "label": "AccountBarrier",
                "name": acc_barrier.get("barrier_type", "Account Barrier"),
                "details": acc_barrier.get("barrier_details", ""),
                "type": "barrier",
                "status": barrier_status
            })
            edges.append({
                "source": "node_account",
                "target": "node_barrier",
                "label": "HAS_HISTORICAL_BARRIER",
                "status": "highlighted"
            })

            if is_case_1:
                source_node = "node_hcp" if hcp_name else "node_rep"
                edges.append({
                    "source": source_node,
                    "target": "node_barrier",
                    "label": "RAISED_BARRIER (Case 1)",
                    "status": "case_1_active"
                })
            elif is_case_2:
                edges.append({
                    "source": "node_rep",
                    "target": "node_barrier",
                    "label": "PROBED_BY_BOT (Case 2)",
                    "status": "case_2_probed"
                })

            # Account Request node
            if is_case_1 or is_case_2 or any(w in hist_text for w in ["barrier", "flag", "biomarker", "handoff", "prior auth", "turnaround"]):
                ask_text = acc_barrier.get("account_ask") or acc_barrier.get("barrier_details", "")
                short_ask = (ask_text[:36] + "...") if len(ask_text) > 36 else ask_text
                nodes.append({
                    "id": "node_request",
                    "label": "AccountRequest",
                    "name": short_ask,
                    "type": "request",
                    "status": "active"
                })
                edges.append({
                    "source": "node_barrier",
                    "target": "node_request",
                    "label": "REQUESTED_SOLUTION",
                    "status": "active"
                })

        # Node: Action / Closure (if timing or action items discussed)
        has_action = any(w in hist_text for w in ["next step", "suitability", "medical history", "couple of weeks", "next week", "follow-up", "wrap", "close"])
        if has_action:
            action_name = "Review Suitability & Follow-up" if role_upper == "OS" else "Reimbursement Action Plan"
            nodes.append({
                "id": "node_action",
                "label": "NextAction",
                "name": action_name,
                "type": "action",
                "status": "active"
            })
            edges.append({
                "source": "node_rep",
                "target": "node_action",
                "label": "AGREED_ACTION",
                "status": "active"
            })

        # 3. Generate Rich Mermaid Entity-Relationship Diagram
        mmd_lines = ["flowchart TD"]
        mmd_lines.append("  classDef rep fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#312E81;")
        mmd_lines.append("  classDef account fill:#ECFDF5,stroke:#10B981,stroke-width:2px,color:#065F46;")
        mmd_lines.append("  classDef hcp fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E40AF;")
        mmd_lines.append("  classDef brand fill:#FEF2F2,stroke:#EF4444,stroke-width:2px,color:#991B1B;")
        mmd_lines.append("  classDef cohort fill:#F5F3FF,stroke:#8B5CF6,stroke-width:2px,color:#5B21B6;")
        mmd_lines.append("  classDef topic fill:#F8FAFC,stroke:#64748B,stroke-width:2px,color:#334155;")
        mmd_lines.append("  classDef barrierCase1 fill:#FEF3C7,stroke:#D97706,stroke-width:3px,color:#92400E;")
        mmd_lines.append("  classDef barrierCase2 fill:#FFEDD5,stroke:#EA580C,stroke-width:3px,color:#9A3412;")
        mmd_lines.append("  classDef barrierPending fill:#F8FAFC,stroke:#CBD5E1,stroke-width:1px,stroke-dasharray: 4 4,color:#64748B;")
        mmd_lines.append("  classDef request fill:#FEF9C3,stroke:#CA8A04,stroke-width:2px,color:#713F12;")
        mmd_lines.append("  classDef action fill:#FDF4FF,stroke:#C026D3,stroke-width:2px,color:#701A75;")

        for n in nodes:
            nid = n["id"]
            name = n["name"].replace('"', "'")
            ntype = n["type"]
            st = n["status"]
            
            if ntype == "rep":
                mmd_lines.append(f'  {nid}["👤 {name}"]:::rep')
            elif ntype == "role":
                mmd_lines.append(f'  {nid}["💼 {name}"]:::rep')
            elif ntype == "account":
                mmd_lines.append(f'  {nid}[("🏥 {name}")]:::account')
            elif ntype == "hcp":
                mmd_lines.append(f'  {nid}(["👨‍⚕️ {name}"]):::hcp')
            elif ntype == "brand":
                mmd_lines.append(f'  {nid}[/"💊 {name}"/]:::brand')
            elif ntype == "cohort":
                mmd_lines.append(f'  {nid}{{"🎯 {name}"}}:::cohort')
            elif ntype == "topic":
                mmd_lines.append(f'  {nid}["📋 {name}"]:::topic')
            elif ntype == "barrier":
                cls = "barrierCase1" if st == "case_1_active" else ("barrierCase2" if st == "case_2_probed" else "barrierPending")
                badge = "⚠️ Case 1: " if st == "case_1_active" else ("💡 Case 2: " if st == "case_2_probed" else "⚠️ ")
                mmd_lines.append(f'  {nid}{{"{badge}{name}"}}:::{cls}')
            elif ntype == "request":
                mmd_lines.append(f'  {nid}>"📝 {name}"]:::request')
            elif ntype == "action":
                mmd_lines.append(f'  {nid}[("✅ {name}")]:::action')

        for e in edges:
            src = e["source"]
            tgt = e["target"]
            lbl = e["label"]
            mmd_lines.append(f'  {src} -->|"{lbl}"| {tgt}')

        mermaid_str = "\n".join(mmd_lines)

        return {
            "nodes": nodes,
            "edges": edges,
            "mermaid": mermaid_str,
            "barrier_case": effective_case if acc_barrier else None
        }

    def _index_graph(self):
        """Indexes all graph nodes and edges for sub-millisecond retrieval."""
        # Index Nodes
        for node in self.kg_data.get("nodes", []):
            nid = node["id"]
            label = node["label"]
            props = node.get("properties", {})
            
            if label == "Role":
                role_name = props.get("name")
                self.roles[role_name] = props
            elif label == "Domain":
                self.domains[nid] = props
            elif label == "Topic":
                tname = props.get("name")
                self.topics[tname] = props
            elif label == "Account":
                self.accounts[props.get("name", nid)] = props
            elif label == "AccountBarrier":
                if props not in self.account_barriers:
                    self.account_barriers.append(props)
            elif label == "ComplianceRule":
                self.compliance_rules.append(props)
            elif label == "CoreResponsibility":
                if "OS" in nid or "os" in nid.lower():
                    self.core_responsibilities["OS"].append(props)
                elif "FRM" in nid:
                    self.core_responsibilities["FRM"].append(props)
            elif label == "ExclusionRule":
                if "OS" in nid or "os" in nid.lower():
                    self.exclusion_rules["OS"].append(props)
                elif "FRM" in nid:
                    self.exclusion_rules["FRM"].append(props)

        # Index Edges
        for edge in self.kg_data.get("edges", []):
            src = edge["source"]
            tgt = edge["target"]
            etype = edge["type"]
            
            if etype == "HAS_IN_SCOPE_TOPIC":
                role = src.replace("role:", "")
                topic = tgt.replace("topic:", "").replace("_", " ")
                if role in self.in_scope_topics:
                    self.in_scope_topics[role].add(topic)
            elif etype == "HAS_OUT_OF_SCOPE_TOPIC":
                role = src.replace("role:", "")
                topic = tgt.replace("topic:", "").replace("_", " ")
                if role in self.out_of_scope_topics:
                    self.out_of_scope_topics[role].add(topic)
            elif etype == "COLLABORATES_WITH":
                role = src.replace("role:", "")
                target_role = tgt.replace("role:", "")
                proto = edge.get("properties", {}).get("protocol", "")
                if role in self.collaborations:
                    self.collaborations[role].append({
                        "target_role": target_role,
                        "protocol": proto
                    })
            elif etype == "ABOUT_TOPIC":
                resp_id = src
                topic_name = tgt.replace("topic:", "").replace("_", " ")
                role = "OS" if "OS" in resp_id else ("FRM" if "FRM" in resp_id else None)
                if role:
                    self.topic_to_resp[topic_name][role].append(resp_id)

        # Enforce distinct in-scope vs out-of-scope guarantee
        for role in ["OS", "FRM"]:
            self.out_of_scope_topics[role] = self.out_of_scope_topics[role] - self.in_scope_topics[role]

    def get_role_context(self, role: str) -> Dict[str, Any]:
        """Returns the high-level ontological profile and boundary for a persona."""
        role_upper = role.upper()
        role_info = self.roles.get(role_upper, {})
        return {
            "role_code": role_upper,
            "full_name": role_info.get("full_name", f"Field Role {role_upper}"),
            "description": role_info.get("description", ""),
            "primary_domain": role_info.get("primary_domain", ""),
            "allowed_topics": sorted(list(self.in_scope_topics.get(role_upper, set()))),
            "forbidden_topics": sorted(list(self.out_of_scope_topics.get(role_upper, set()))),
            "exclusion_rules": [e.get("text") for e in self.exclusion_rules.get(role_upper, [])]
        }

    def detect_covered_topics(self, conversation_history: Optional[List[Dict[str, str]]], latest_answer: str = "") -> Set[str]:
        """
        Dynamically analyzes the entire conversation history and latest user answer to identify
        all Knowledge Graph topics that have already been discussed or resolved.
        """
        covered = set()
        full_text = " ".join([h.get("text", "") for h in (conversation_history or [])]) + " " + latest_answer
        text_lower = full_text.lower()
        
        # 1. Account / Stakeholder identification
        if any(w in text_lower for w in ["dr.", "dr ", "doctor", "hospital", "clinic", "center", "centre", "urology", "oncology", "apollo", "met with", "spoke with"]):
            covered.add("account identification")
            covered.add("stakeholder management")
            
        # 2. Product Efficacy & Indication
        if any(w in text_lower for w in ["nmibc", "bladder", "bcg", "post bcg", "bcg failure", "efficacy", "clinical trial", "pfs", "mariposa", "egfr", "exon 20", "indication", "eligible patient"]):
            covered.add("efficacy safety product info")
            
        # 3. Treatment Sequencing & Pathway
        # Only mark covered if clinical status and suitability for treatment have actually been discussed!
        if any(w in text_lower for w in ["suitable for inlexzo", "candidate suitable", "decide the treatment", "treatment approach in next", "check the complete medical history"]):
            covered.add("treatment sequencing")
            
        # 4. Clinical Operational Workflow
        if any(w in text_lower for w in ["workflow", "room setup", "anatomical demo", "insertion", "device", "refresher", "nursing staff", "app team", "order set", "ehr"]):
            covered.add("clinical operational workflow")
            
        # 5. Dosing & Administration
        if any(w in text_lower for w in ["dosing", "infusion", "administration", "subcutaneous", "reconstitution", "pre-med", "dose"]):
            covered.add("dosing administration")
            
        # 6. Prior Authorization & Payer Coverage
        if any(w in text_lower for w in ["prior auth", "prior authorization", "pa ", "payer", "coverage", "appeal", "denial", "formulary"]):
            covered.add("prior authorization")
            covered.add("payer coverage")
            
        # 7. Affordability & Reimbursement
        if any(w in text_lower for w in ["copay", "co-pay", "pap", "patient assistance", "hub", "specialty pharmacy", "reimbursement", "billing", "claim"]):
            covered.add("affordability patient support")
            covered.add("reimbursement")

        # 8. Barriers discussed or explicitly denied
        if any(w in text_lower for w in ["barrier", "barriers", "no we havent discussed about this", "no there arent any", "no barriers", "none"]):
            covered.add("patient access")
            
        # 9. Next Steps / Follow-up Planning
        if any(w in text_lower for w in ["follow-up", "next couple of weeks", "next week", "email", "schedule", "decide the treatment approcah"]):
            covered.add("territory engagement planning")
            
        return covered

    def get_dynamic_target_topic(
        self,
        role: str,
        current_state: str,
        conversation_history: Optional[List[Dict[str, str]]],
        candidate_answer: str = "",
        detected_topics: Optional[List[str]] = None,
        covered_topics: Optional[Set[str]] = None
    ) -> str:
        """
        Dynamically selects the next best Knowledge Graph target topic based on:
        1. What the user just said in candidate_answer.
        2. What has already been discussed in cumulative conversation history.
        3. The unaddressed roadmap topics for this role in the Knowledge Graph.
        
        Guarantees:
        - NEVER targets a topic that has already been thoroughly covered or answered.
        - NEVER repeats purpose or barrier questions if already discussed.
        """
        role_upper = role.upper()
        detected = detected_topics or []
        ans_lower = candidate_answer.lower()
        hist_text = " ".join([h.get("text", "") for h in (conversation_history or [])]) + " " + candidate_answer
        hist_lower = hist_text.lower()
        
        # Build comprehensive covered topics set
        all_covered = set(covered_topics or set())
        all_covered.update(self.detect_covered_topics(conversation_history, candidate_answer))
        
        # Check if user just stated doctor/account in candidate_answer
        just_identified_account = any(w in ans_lower for w in ["dr.", "dr ", "doctor", "at apollo", "at hospital", "clinic", "center", "group"])
        if just_identified_account:
            all_covered.add("account identification")
            
        # Check if user just stated purpose or indication
        just_stated_purpose = any(w in ans_lower for w in ["to get an update", "main purpose", "eligible patient", "nmibc", "bladder", "treatment pathway", "bcg failure", "efficacy", "reimbursement check-in", "prior auth"])
        if just_stated_purpose:
            all_covered.add("account identification")
            if role_upper == "FRM":
                all_covered.add("patient access")
                all_covered.add("payer coverage")

        # Check if user just answered barrier question with "No"
        if any(w in ans_lower for w in ["no we havent", "no there arent", "no barriers", "none", "not discussed", "nothing"]):
            all_covered.add("patient access")

        # Check if treatment sequencing is resolved (BCG status and next step stated)
        if any(w in hist_lower for w in ["medical history", "suitable for inlexzo", "treatment approach in next"]):
            all_covered.add("treatment sequencing")
            
        # Roadmap for role
        roadmap = self.role_topic_roadmap.get(role_upper, [])
        unaddressed = [t for t in roadmap if t not in all_covered]
        
        # If user directly raised an in-scope topic in latest utterance that needs immediate follow-up
        if detected:
            prior_covered = self.detect_covered_topics(conversation_history)
            if "efficacy safety product info" in detected and any(w in ans_lower for w in ["efficacy", "pfs", "trial", "survival", "data"]):
                if "efficacy safety product info" not in prior_covered:
                    return "efficacy safety product info"
            for top in detected:
                if top in self.in_scope_topics.get(role_upper, set()):
                    if top not in prior_covered:
                        return top

        # If user mentioned eligible patient / BCG failure / treatment pathway -> treatment sequencing
        if any(w in ans_lower for w in ["eligible patient", "bcg failure", "post bcg", "bcg-unresponsive", "treatment pathway"]):
            if "treatment sequencing" not in all_covered:
                return "treatment sequencing"

        # If user mentioned medical history / checking suitability -> patient access or territory planning
        if any(w in ans_lower for w in ["suitable for inlexzo", "check the complete medical history"]):
            if "patient access" not in all_covered:
                return "patient access"
            elif "territory engagement planning" not in all_covered:
                return "territory engagement planning"

        # Otherwise pick the first unaddressed topic from the Knowledge Graph roadmap
        if unaddressed:
            return unaddressed[0]
            
        # Fallback to next steps / wrap-up if roadmap is complete
        return "territory engagement planning"

    def retrieve_kg_node_inquiry(
        self,
        role: str,
        target_topic: str,
        conversation_history=None,
        candidate_answer: str = "",
        brand: Optional[str] = None,
        hcp: str = "the doctor",
        account: Optional[str] = None
    ) -> str:
        """
        Dynamically analyzes what the user just said in `candidate_answer` and the dialogue history
        to generate the exact next clinical inquiry grounded in the J&J Commercial Oncology Knowledge Graph.
        Guarantees:
        - Strict adherence to what the user actually stated in their response.
        - Strict anchoring to the active HCP name, Account, and Brand (zero hallucinated doctor names).
        - Account Barrier Intelligence: Conditions inquiries on known historical friction at major accounts.
        - Zero redundant questions for information already shared in dialogue history.
        - Natural progression across J&J Knowledge Graph commercial oncology duties.
        """
        role_upper = (role or "OS").upper()
        raw_hcp = (hcp or "the doctor").strip()
        if raw_hcp and raw_hcp.lower() != "the doctor":
            if not raw_hcp.lower().startswith("dr.") and not raw_hcp.lower().startswith("dr "):
                hcp_name = f"Dr. {raw_hcp}"
            else:
                # Ensure standard formatting like "Dr. Name"
                hcp_name = re.sub(r'^(dr\.?)\s*', 'Dr. ', raw_hcp, flags=re.IGNORECASE)
        else:
            hcp_name = "the doctor"
        if hcp_name == "the doctor":
            doc_m = re.search(r'\b(dr\.?\s+[A-Z][a-z]+)\b', candidate_answer)
            if doc_m:
                hcp_name = doc_m.group(1).title()
        prior_ai = [h.get("text", "").lower() for h in (conversation_history or []) if h.get("speaker") == "AI"]
        cand_lower = (candidate_answer or "").strip().lower()
        hist_text = " ".join([h.get("text", "") for h in (conversation_history or [])]).lower()
        has_known_hcp = any(w in hist_text or w in cand_lower for w in ["dr.", "dr ", "doctor"])

        default_role_brand = "INLEXZO" if role_upper == "OS" else "RYBREVANT"
        if any(w in (cand_lower + " " + hist_text) for w in ["nmibc", "bladder", "bcg", "anurag", "apollo"]) or (brand and "inlexzo" in brand.lower()):
            brand_name = "INLEXZO"
        elif any(w in (cand_lower + " " + hist_text) for w in ["nsclc", "lung", "egfr", "exon 20", "amivantamab", "lazcluze"]) or (brand and "rybrevant" in brand.lower()):
            brand_name = "RYBREVANT"
        else:
            brand_name = brand or default_role_brand

        resolved_account = account
        is_canonical = any(c[1].lower() in (resolved_account or "").lower() for c in CANONICAL_ACCOUNTS)
        if not is_canonical:
            full_context = cand_lower + " " + hist_text
            for k, canonical in CANONICAL_ACCOUNTS:
                if k in full_context:
                    resolved_account = canonical
                    break

        acc_barrier = self.get_account_barrier(resolved_account, role_upper)
        self.last_barrier_case = None

        # Dual-Case Barrier Check:
        # Case 1: If user introduces or asks about the barrier
        barrier_mention = self.detect_barrier_mention(cand_lower, resolved_account, role_upper)
        if barrier_mention:
            case_1_q = barrier_mention.get("case_1_inquiry")
            if case_1_q:
                formatted_case1 = case_1_q.format(hcp=hcp_name, brand=brand_name)
                # Ensure we NEVER re-ask the barrier question if it was already asked in prior_ai
                if not any(formatted_case1.lower() in q.lower() or q.lower() in formatted_case1.lower() for q in prior_ai):
                    self.last_barrier_case = "CASE_1_USER_INITIATED"
                    return formatted_case1

        # Cross-functional collaboration trigger check (e.g. unsolicited scientific inquiry -> MSL / MIR)
        collab = self.check_collaboration_triggers(role_upper, cand_lower)
        if collab and "MSL" in collab.get("target_role", ""):
            return f"Did you log a Medical Information Request (MIR) or loop in the MSL for {hcp_name}?"
        elif collab and "FRM" in collab.get("target_role", ""):
            return f"Did you connect the office with their designated Field Reimbursement Manager (FRM) for {brand_name}?"

        if role_upper == "OS":
            # 0. User acknowledging greeting or hasn't identified doctor/account yet
            if not has_known_hcp and not any(w in cand_lower for w in ["dr.", "dr ", "doctor"]):
                return "Who did you meet with, and where?"

            # 1. User answered Who/Where (Account Identification) -> Immediate Barrier Validation
            if any(w in cand_lower for w in ["dr.", "dr ", "doctor", "urology", "hospital", "clinic", "center", "atlantic", "capital", "northside", "summit", "valley", "temple", "regional", "central ohio"]) and not any("discussion" in q or "purpose" in q for q in prior_ai):
                display_hcp = hcp_name if (has_known_hcp or (hcp_name and hcp_name != "the doctor")) else None
                # Inject role-specific account barrier validation into the first substantive question
                if acc_barrier and resolved_account:
                    self.last_barrier_case = "CASE_2_PROACTIVE_FOLLOWUP"
                    case_2_q = acc_barrier.get("case_2_inquiry")
                    if case_2_q:
                        formatted_q = case_2_q.format(hcp=display_hcp or "the doctor", brand=brand_name)
                        if display_hcp:
                            return f"What was the main {brand_name} discussion with {display_hcp} today? {formatted_q}"
                        return f"What was the main {brand_name} discussion today? {formatted_q}"
                # No barrier registered — fall back to standard question
                if display_hcp:
                    return f"What was the main {brand_name} discussion with {display_hcp} today?"
                return f"What was the main {brand_name} discussion today?"

            # 2a. User answered that HCP has no eligible patients
            if any(w in cand_lower for w in ["no eligible", "no patients", "no patient", "haven't identified", "none right now", "no case", "do not have", "don't have"]):
                if not any("alternative" in q or "protocol" in q for q in prior_ai):
                    return f"What alternative treatment approach or protocol is {hcp_name} currently following?"
                elif not any("support" in q for q in prior_ai):
                    return f"Did {hcp_name} mention any support or resources needed?"
                elif not any("action items" in q or "follow-up" in q for q in prior_ai):
                    return "Any follow-up or action items from the call?"
                return "Any other account context to capture?"

            # 2b. User answered Call Purpose / Seeking update on eligible patients or treatment pathway
            if any(w in cand_lower for w in ["to get an update", "eligible patient", "treatment pathway", "bladder cancer", "nmibc"]):
                if not any(w in cand_lower for w in ["identified", "bcg failure", "unresponsive", "found a patient"]):
                    if not any("treatment sequencing" in q or "pathway protocol" in q for q in prior_ai):
                        return f"What specific treatment sequencing or pathway protocol did {hcp_name} discuss for eligible {brand_name} candidates?"

            # 3. User answered that a patient was identified with BCG failure
            if any(w in cand_lower for w in ["bcg failure", "post bcg", "patient has been identified", "found a patient", "eligible case identified"]):
                if not any(w in cand_lower for w in ["unresponsive", "completed full course"]):
                    return "What is the current BCG status for that case?"

            # 4. User answered BCG status (unresponsive / completed full course)
            if any(w in cand_lower for w in ["unresponsive", "completed full course", "intolerant", "refractory"]):
                return "What is the next step for the patient?"

            # 4b. User mentioned waiting for pathology, biopsy, or diagnostic test results
            if any(w in cand_lower for w in ["pathology", "biopsy", "lab results", "test results", "waiting for results", "confirm high-grade"]):
                if not any("pathology" in q or "results" in q for q in prior_ai):
                    return "When are the pathology results expected for that patient?"

            # 5. User answered Next Step / Suitability check / Patient history evaluation
            # Context-aware: generate follow-up based on what the user actually said
            if any(w in cand_lower for w in ["medical history", "patient history", "suitable for inlexzo", "suitable for", "sutable for", "evaluate", "evaluating", "whether he is eligible", "eligibility", "suitability", "full details"]):
                # Ask about the specific clinical evaluation the HCP is performing
                if any(w in cand_lower for w in ["patient history", "medical history", "full details", "evaluating", "whether he is eligible", "eligibility"]):
                    if not any("clinical parameters" in q or "evaluation criteria" in q or "clinical factors" in q for q in prior_ai):
                        return f"What specific clinical parameters or medical history factors is {hcp_name} evaluating for {brand_name} eligibility?"
                # Ask about access or coverage barriers
                if not any("barrier" in q or "access" in q or "coverage" in q for q in prior_ai):
                    return f"Did {hcp_name} anticipate any access or coverage considerations for {brand_name}?"

            # 6. User answered negatively / dismissively / topic not discussed
            # Covers: "no", "none", "no barriers", "as of now there are no barriers",
            #         "we havent discussed about this", "not discussed", "didnt discuss", etc.
            negative_starts = ["no", "none", "not really", "there arent", "there aren't", "as of now", "we havent", "we haven't", "haven't discussed", "havent discussed", "not discussed", "didnt discuss", "didn't discuss", "we didnt", "we didn't"]
            negative_exact = ["no", "none", "no barriers", "none as of now", "no barrier"]
            is_negative = any(cand_lower.startswith(w) for w in negative_starts) or cand_lower.strip() in negative_exact or any(w in cand_lower for w in ["no barriers", "no barrier", "havent discussed", "haven't discussed", "not discussed", "as of now there are no"])
            if is_negative:
                if not any("timing" in q or "touchpoint" in q or "scheduled" in q for q in prior_ai):
                    return f"What is the target timing or next scheduled touchpoint with {hcp_name} for {brand_name}?"
                elif not any("support" in q or "resource" in q for q in prior_ai):
                    return f"Did {hcp_name} mention any support or resources needed?"
                elif not any("action items" in q or "follow-up" in q for q in prior_ai):
                    return "Any follow-up or action items from the call?"
                elif not any("account context" in q for q in prior_ai):
                    return "Any other account context to capture?"
                else:
                    return "I have enough for the call note. Should we wrap here?"

            # 7. User answered Timing ("couple of weeks", "next week", "next month", "weeks", "month", "days")
            if any(w in cand_lower for w in ["couple of weeks", "next week", "next month", "weeks", "month", "days"]):
                if not any("support" in q for q in prior_ai):
                    return f"Did {hcp_name} mention any support or resources needed?"
                elif not any("action items" in q or "follow-up" in q for q in prior_ai):
                    return "Any follow-up or action items from the call?"
                elif not any("account context" in q for q in prior_ai):
                    return "Any other account context to capture?"
                return "I have enough for the call note. Should we wrap here?"

            # 8. User answered Support needed ("nothing at the moment", "no", "not right now")
            if any(w in cand_lower for w in ["not mention any thing", "nothing at the moment", "not at the moment", "no support", "nothing"]):
                return "Any follow-up or action items from the call?"

            # 9. User answered Follow-Up Action Items ("stay in touch", "follow up", "email", "check back")
            if any(w in cand_lower for w in ["stay in touch", "saty in touch", "follow up", "follow-up", "check back", "get the updated"]):
                return "Any other account context to capture?"

            # 10. User answered Account Context ("this is all", "nothing else", "that's all", "that is all")
            if any(w in cand_lower for w in ["this is all", "nothing else", "that's all", "that is all", "all we have"]):
                return "I have enough for the call note. Should we wrap here?"

            # 11. User confirmed wrap up ("yes", "sure", "wrap it", "close it")
            if cand_lower in ["yes", "yeah", "sure", "wrap it", "done", "ok", "okay", "yep"]:
                return f"Thank you, the call notes for {hcp_name} have been captured and logged compliantly. Session closed."

            # Semantic Intent Matching via Embedding KG for any unscripted user phrasing
            if getattr(self, "embedder", None):
                matched_intent = self.embedder.match_conversation_intent(cand_lower, role="OS", threshold=0.40)
                if matched_intent:
                    tmpl = matched_intent.get("inquiry_template")
                    if tmpl:
                        q_gen = tmpl.format(hcp=hcp_name, brand=brand_name)
                        if not any(q_gen.lower() in q.lower() or q.lower() in q_gen.lower() for q in prior_ai):
                            return q_gen

            # Defer to Dynamic Knowledge Graph SLM generation for all other conversational / clinical topics
            return None

        else: # FRM
            if target_topic == "account identification":
                return "Who did you meet with, and where was the account?"
            if "prior authorization" not in cand_lower and not any("prior authorization" in q for q in prior_ai):
                if acc_barrier and resolved_account:
                    self.last_barrier_case = "CASE_2_PROACTIVE_FOLLOWUP"
                    if "Apollo" in resolved_account:
                        return f"Did the team at {resolved_account} raise any concerns regarding prior authorization turnaround times or request approved access-support resources?"
                    elif "Fortis" in resolved_account:
                        return f"Did patient cost exposure come up, or did {resolved_account} ask about copay and benefits verification resources?"
                    elif "Manipal" in resolved_account:
                        return f"Did {resolved_account} ask for support on typical payer requirements to reduce prior authorization back-and-forth?"
                    elif "Max" in resolved_account:
                        return f"Did {resolved_account} discuss the formulary exception pathway or committee timing for {brand_name}?"
                    elif "Narayana" in resolved_account:
                        return f"Did {resolved_account} have questions regarding recent payer policy updates or approved access support channels for {brand_name}?"

            # Semantic Intent Matching via Embedding KG for FRM unscripted phrasing
            if getattr(self, "embedder", None):
                matched_intent = self.embedder.match_conversation_intent(cand_lower, role="FRM", threshold=0.40)
                if matched_intent:
                    tmpl = matched_intent.get("inquiry_template")
                    if tmpl:
                        q_gen = tmpl.format(hcp=hcp_name, brand=brand_name)
                        if not any(q_gen.lower() in q.lower() or q.lower() in q_gen.lower() for q in prior_ai):
                            return q_gen

            # Defer to Dynamic Knowledge Graph SLM generation for all reimbursement and access topics
            return None

    def retrieve_relevant_duties(self, role: str, target_topic: str, max_duties: int = 2, candidate_answer: str = "") -> List[str]:
        """Retrieves specific J&J CoreResponsibility text from the Knowledge Graph for the active topic."""
        role_upper = role.upper()
        relevant = []
        
        # 1. Semantic embedding retrieval if candidate_answer provided
        if candidate_answer and getattr(self, "embedder", None):
            sem_duties = self.embedder.match_relevant_responsibilities(candidate_answer, role=role_upper, top_k=max_duties)
            for d in sem_duties:
                if d.get("text") and d["text"] not in relevant:
                    relevant.append(d["text"])
            if len(relevant) >= max_duties:
                return relevant[:max_duties]

        # 2. Look for duties mentioning the target topic or linked in graph
        for resp in self.core_responsibilities.get(role_upper, []):
            text = resp.get("text", "")
            topics = [t.lower() for t in resp.get("topics", [])]
            if target_topic.lower() in topics or any(k in text.lower() for k in target_topic.split()):
                if text not in relevant:
                    relevant.append(text)
                if len(relevant) >= max_duties:
                    break
                    
        # Fallback to general responsibilities if topic-specific not found
        if not relevant and self.core_responsibilities.get(role_upper):
            relevant.append(self.core_responsibilities[role_upper][0].get("text", ""))
            
        return relevant[:max_duties]

    def compute_unaddressed_topics(self, role: str, covered_topics: Set[str]) -> List[str]:
        """Computes which mandatory topics on this role's roadmap have not yet been addressed."""
        role_upper = role.upper()
        roadmap = self.role_topic_roadmap.get(role_upper, [])
        unaddressed = [t for t in roadmap if t not in covered_topics]
        return unaddressed

    def check_collaboration_triggers(self, role: str, utterance: str) -> Optional[Dict[str, str]]:
        """Detects if cross-functional handoff keywords are triggered in the utterance."""
        role_upper = role.upper()
        text_lower = utterance.lower()
        
        if role_upper == "OS":
            if any(w in text_lower for w in ["biomarker", "molecular", "exon 20", "unsolicited", "clinical trial data", "off-label"]):
                return {
                    "target_role": "MSL",
                    "reason": "Peer-to-peer scientific exchange / unsolicited biomarker inquiries require Medical Science Liaison (MSL) referral.",
                    "prompt_directive": "Inquire if a Medical Information Request (MIR) or MSL referral is needed."
                }
            if any(w in text_lower for w in ["prior auth", "reimbursement", "denial", "copay", "payer policy"]):
                return {
                    "target_role": "FRM",
                    "reason": "Reimbursement, payer coverage, and prior auth navigation require Field Reimbursement Manager (FRM) support.",
                    "prompt_directive": "Inquire if the account was referred to the local FRM."
                }
        elif role_upper == "FRM":
            if any(w in text_lower for w in ["clinical trial", "pfs", "overall survival", "efficacy comparison"]):
                return {
                    "target_role": "OS / MSL",
                    "reason": "Clinical trial efficacy discussions are out-of-scope for FRM and belong to OS or MSL.",
                    "prompt_directive": "Note that clinical topics cannot be captured in FRM notes."
                }
                
        return None

    # ===================================================================
    # FLAT PROMPT FORMAT (aligned with infer.py / merged Llama-3.2-1B)
    # ===================================================================

    @staticmethod
    def build_cumulative_transcript(history: List[Dict[str, str]]) -> Optional[str]:
        """
        Converts a list of {"speaker": "AI"|"User", "text": "..."} dicts
        into a cumulative transcript string: "AI: ...\nUser: ...\nAI: ..."
        Returns None if the history is empty (cold-start / turn 0).
        """
        if not history:
            return None
        lines = []
        for entry in history:
            speaker = entry.get("speaker", "User")
            text = entry.get("text", "")
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def build_flat_row(
        self,
        role: str,
        current_state: str,
        brand: Optional[str] = None,
        persona_name: Optional[str] = None,
        account_name: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        covered_topics: Optional[Set[str]] = None,
        detected_topics: Optional[List[str]] = None,
        candidate_answer: str = ""
    ) -> Dict[str, Any]:
        """
        Produces a complete dynamic row dict matching infer.py's expected schema:
        {
            "role": "OS",
            "brand": "INLEXZO",
            "target_topic": "treatment sequencing",
            "state": "STATE_2_PRIMARY_PURPOSE",
            "kg_context": {
                "allowed_topics": [...],
                "forbidden_topics": [...],
                "applicable_responsibility": "...",
                "compliance_mandate": "..."
            },
            "context": "AI: ...\nUser: ...",
            "persona_name": "Dr Anurag",
            "account_name": "Apollo Hospitals",
            "account_barrier": {...}
        }
        """
        role_upper = role.upper()
        topics = detected_topics or []
        
        # Dynamically compute all covered topics
        dynamic_covered = self.detect_covered_topics(conversation_history, candidate_answer)
        if covered_topics:
            dynamic_covered.update(covered_topics)

        # Auto-detect brand from full conversation context if not explicitly provided
        default_role_brand = "INLEXZO" if role_upper == "OS" else "RYBREVANT"
        resolved_brand = brand
        if not resolved_brand and (conversation_history or candidate_answer):
            history_text = " ".join(e.get("text", "") for e in (conversation_history or [])) + " " + candidate_answer
            detected_b = detect_brand_from_text(history_text)
            if detected_b:
                resolved_brand = detected_b
        resolved_brand = resolved_brand or default_role_brand

        # Auto-detect account if not explicitly provided
        resolved_account = account_name
        is_canonical = any(c[1].lower() in (resolved_account or "").lower() for c in CANONICAL_ACCOUNTS)
        if not is_canonical:
            full_text = " ".join(e.get("text", "") for e in (conversation_history or [])) + " " + candidate_answer
            for k, canonical in CANONICAL_ACCOUNTS:
                if k in full_text.lower():
                    resolved_account = canonical
                    break
        account_barrier = self.get_account_barrier(resolved_account, role_upper)

        role_profile = self.get_role_context(role_upper)
        
        # Dynamically determine the target topic from the Knowledge Graph
        target_topic = self.get_dynamic_target_topic(
            role=role_upper,
            current_state=current_state,
            conversation_history=conversation_history,
            candidate_answer=candidate_answer,
            detected_topics=topics,
            covered_topics=dynamic_covered
        )
        
        relevant_duties = self.retrieve_relevant_duties(role_upper, target_topic, candidate_answer=candidate_answer)
        applicable_responsibility = relevant_duties[0] if relevant_duties else ""

        # Compliance mandate
        compliance_mandate = (
            "Replace 'toxicity' with 'safety concerns'. "
            "Always generate only a single follow-up question. "
            "Never refuse to capture call details. "
            "Do not repeat questions already asked."
        )

        # Cumulative transcript context
        context = self.build_cumulative_transcript(conversation_history) if conversation_history else None

        return {
            "role": role_upper,
            "brand": resolved_brand,
            "target_topic": target_topic,
            "state": current_state,
            "kg_context": {
                "allowed_topics": role_profile["allowed_topics"],
                "forbidden_topics": role_profile["forbidden_topics"],
                "applicable_responsibility": applicable_responsibility,
                "compliance_mandate": compliance_mandate
            },
            "context": context,
            "persona_name": persona_name,
            "account_name": resolved_account,
            "account_barrier": account_barrier,
            "covered_topics": list(dynamic_covered),
            "unaddressed_topics": self.compute_unaddressed_topics(role_upper, dynamic_covered)
        }

    @staticmethod
    def build_flat_user_prompt(row: Dict[str, Any]) -> str:
        """
        Produces the flat user prompt matching infer.py's build_user_prompt().
        This is the exact format the fine-tuned Llama-3.2-1B model expects.
        """
        kg = row["kg_context"]
        lines = [
            f"Brand: {row['brand']}",
            f"Persona: {row['role']}",
        ]
        if row.get("persona_name"):
            lines.append(f"Contact: {row['persona_name']}")
        if row.get("account_name"):
            lines.append(f"Account: {row['account_name']}")
        if row.get("account_barrier"):
            b = row["account_barrier"]
            lines.append(f"Known Account Friction: {b.get('barrier_type')} - {b.get('barrier_details')}")
        lines += [
            f"Target topic: {row['target_topic']}",
            f"Conversation state: {row['state']}",
            f"Allowed topics: {', '.join(kg['allowed_topics'])}",
            f"Forbidden topics: {', '.join(kg['forbidden_topics'])}",
        ]
        if kg.get("applicable_responsibility"):
            lines.append(f"Applicable responsibility: {kg['applicable_responsibility']}")
        if kg.get("compliance_mandate"):
            lines.append(f"Compliance mandate: {kg['compliance_mandate']}")

        context = row["context"] if row.get("context") is not None else "(conversation not yet started)"
        lines.append(f"\nConversation so far:\n{context}")
        return "\n".join(lines)

    def build_flat_prompt_context(
        self,
        role: str,
        current_state: str,
        brand: Optional[str] = None,
        persona_name: Optional[str] = None,
        account_name: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        covered_topics: Optional[Set[str]] = None,
        detected_topics: Optional[List[str]] = None,
        candidate_answer: str = ""
    ) -> Dict[str, Any]:
        """
        Builds the complete flat-format prompt context for SLM inference.
        """
        role_upper = role.upper()
        covered = covered_topics or set()
        topics = detected_topics or []

        # Build the dynamic flat row
        row = self.build_flat_row(
            role=role_upper,
            current_state=current_state,
            brand=brand,
            persona_name=persona_name,
            account_name=account_name,
            conversation_history=conversation_history,
            covered_topics=covered,
            detected_topics=topics,
            candidate_answer=candidate_answer
        )

        user_prompt = self.build_flat_user_prompt(row)
        target_topic = row["target_topic"]
        relevant_duties = self.retrieve_relevant_duties(role_upper, target_topic)
        unaddressed = row.get("unaddressed_topics", [])
        collab_trigger = self.check_collaboration_triggers(role_upper, candidate_answer) if candidate_answer else None

        return {
            "row": row,
            "role": role_upper,
            "current_state": current_state,
            "target_topic": target_topic,
            "account_name": row.get("account_name"),
            "account_barrier": row.get("account_barrier"),
            "relevant_duties": relevant_duties,
            "allowed_topics": row["kg_context"]["allowed_topics"],
            "forbidden_topics": row["kg_context"]["forbidden_topics"],
            "unaddressed_topics": unaddressed,
            "collaboration_trigger": collab_trigger,
            "user_prompt": user_prompt,
            "messages": [
                {"role": "system", "content": FLAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        }


_CACHED_RETRIEVER: Optional[KGContextRetriever] = None

def get_retriever(kg_json_path: str = "kg_os.json") -> KGContextRetriever:
    """Returns a singleton cached instance of KGContextRetriever."""
    global _CACHED_RETRIEVER
    if _CACHED_RETRIEVER is None:
        _CACHED_RETRIEVER = KGContextRetriever(kg_json_path=kg_json_path)
    return _CACHED_RETRIEVER


ROLE_STATE_FLOW = {
    "FRM": [
        "STATE_0_GREETING_INITIATION",
        "STATE_1_ACCOUNT_STAKEHOLDER",
        "STATE_2_PRIMARY_PURPOSE",
        "STATE_3A_PRIOR_AUTH_PAYER",
        "STATE_3B_AFFORDABILITY_COPAY_PAP",
        "STATE_3C_HUB_SPECIALTY_PHARMACY",
        "STATE_6_NEXT_ACTIONS",
        "STATE_7_WRAP_UP_CONFIRMATION",
    ],
    "OS": [
        "STATE_0_GREETING_INITIATION",
        "STATE_1_ACCOUNT_STAKEHOLDER",
        "STATE_2_PRIMARY_PURPOSE",
        "STATE_3E_WORKFLOW_DEMO_REFRESHER",
        "STATE_4_BARRIERS_LOGISTICS",
        "STATE_5_CROSS_FUNCTIONAL_HANDOFF",
        "STATE_6_NEXT_ACTIONS",
        "STATE_7_WRAP_UP_CONFIRMATION",
    ],
}

def get_state_by_turn(role: str, turn_index: int = 0) -> str:
    flow = ROLE_STATE_FLOW.get(role.upper(), ROLE_STATE_FLOW["FRM"])
    if turn_index < len(flow):
        return flow[turn_index]
    return flow[-1]


