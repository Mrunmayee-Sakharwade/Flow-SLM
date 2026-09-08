"""
Knowledge Graph Rule Engine (Deterministic Scope & Compliance Guardrail)
========================================================================
Implements 100% deterministic rule enforcement using the Persona Solid Cancer
Knowledge Graph (OS/FRM scoped JSON and Neo4j ontology).

Functions:
  - Scope Validation: Detects if a role discusses an out-of-scope topic.
  - Exclusion Rule Evaluation: Maps violations directly to graph ExclusionRule nodes.
  - Compliance Rule Evaluation: Detects PHI/PII, MIR, and Toxicity policy violations.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional

class KGRuleEngine:
    def __init__(self, kg_json_path: str = "kg_os.json"):
        with open(kg_json_path, "r", encoding="utf-8") as f:
            self.kg_data = json.load(f)
        
        self.roles = {}
        self.topics = {}
        self.in_scope_topics = {"OS": set(), "FRM": set()}
        self.out_of_scope_topics = {"OS": set(), "FRM": set()}
        self.exclusion_rules = {"OS": [], "FRM": []}
        self.compliance_rules = []
        self.core_responsibilities = {"OS": [], "FRM": []}
        
        self.enable_phi_guardrail = os.getenv("ENABLE_PHI_GUARDRAIL", "false").lower() in ["true", "1", "yes"]
        self.enable_mir_guardrail = os.getenv("ENABLE_MIR_GUARDRAIL", "false").lower() in ["true", "1", "yes"]
        
        # Load Account Barrier Intelligence
        self.account_barriers = []
        barriers_path = "account_barriers.json"
        if not os.path.isabs(barriers_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cand = os.path.join(base_dir, barriers_path)
            if os.path.exists(cand):
                barriers_path = cand
        if os.path.exists(barriers_path):
            try:
                with open(barriers_path, "r", encoding="utf-8") as bf:
                    self.account_barriers = json.load(bf)
            except Exception as e:
                pass

        self._index_graph()
    
    def _index_graph(self):
        # Index Nodes
        for node in self.kg_data.get("nodes", []):
            nid = node["id"]
            label = node["label"]
            props = node.get("properties", {})
            
            if label == "Role":
                role_name = props.get("name")
                self.roles[role_name] = node
            elif label == "Topic":
                topic_name = props.get("name")
                self.topics[topic_name] = node
            elif label == "ComplianceRule":
                self.compliance_rules.append(node)
            elif label == "ExclusionRule":
                if "OS" in nid:
                    self.exclusion_rules["OS"].append(node)
                elif "FRM" in nid:
                    self.exclusion_rules["FRM"].append(node)
            elif label == "CoreResponsibility":
                if "OS" in nid:
                    self.core_responsibilities["OS"].append(node)
                elif "FRM" in nid:
                    self.core_responsibilities["FRM"].append(node)
        
        # Index Topic Scopes from Edges
        for edge in self.kg_data.get("edges", []):
            etype = edge.get("type") or edge.get("label")
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            
            role = "OS" if "os" in src.lower() else "FRM"
            topic_name = tgt.replace("topic:", "").replace("_", " ")
            if etype == "HAS_IN_SCOPE_TOPIC":
                self.in_scope_topics[role].add(topic_name)
            elif etype in ["HAS_OUT_OF_SCOPE_TOPIC", "FORBIDS_TOPIC", "EXCLUDES_TOPIC"]:
                self.out_of_scope_topics[role].add(topic_name)
                
        # Guarantee: In-scope topics cannot be treated as out-of-scope
        for role in ["OS", "FRM"]:
            self.out_of_scope_topics[role] = self.out_of_scope_topics[role] - self.in_scope_topics[role]

    def evaluate_compliance(self, text: str) -> Dict[str, Any]:
        """
        Evaluates global compliance rules.
        Note: All compliance error interruptions are completely turned OFF by default.
        """
        enable_compliance = os.getenv("ENABLE_COMPLIANCE_GUARDRAIL", "false").lower() in ["true", "1", "yes"]
        results = {
            "is_compliant": True,
            "phi_detected": False,
            "toxicity_detected": False,
            "mir_detected": False,
            "redacted_text": text,
            "actions": []
        }
        
        if not enable_compliance:
            return results

        # 1. PHI / PII Check (When explicitly enabled)
        if self.enable_phi_guardrail:
            phi_patterns = [
                r"\b(patient\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b",
                r"\b(DOB|date\s+of\s+birth)\s*[:=]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                r"\b(MRN|medical\s+record)\s*#?\s*\d+\b",
                r"\b(SSN|social\s+security)\s*#?\s*\d{3}-\d{2}-\d{4}\b"
            ]
            for pat in phi_patterns:
                if re.search(pat, text, re.IGNORECASE):
                    results["is_compliant"] = False
                    results["phi_detected"] = True
                    results["actions"].append({
                        "rule_id": "rule:privacy_phi_pii",
                        "type": "PHI_VIOLATION",
                        "severity": "mandatory",
                        "action": "redact",
                        "message": "Protected Health Information (PHI/PII) detected. Details must be redacted from call notes."
                    })
                    results["redacted_text"] = re.sub(pat, "[REDACTED_PHI]", results["redacted_text"], flags=re.IGNORECASE)
        
        # 2. Toxicity Word Control
        if re.search(r"\btoxicity\b", text, re.IGNORECASE):
            results["toxicity_detected"] = True
            results["redacted_text"] = re.sub(r"\btoxicity\b", "safety concerns", results["redacted_text"], flags=re.IGNORECASE)
            
        # 3. MIR Handling
        if self.enable_mir_guardrail:
            if re.search(r"\b(unsolicited\s+request|off-label\s+question|medical\s+information\s+request|mir)\b", text, re.IGNORECASE):
                results["mir_detected"] = True
                results["actions"].append({
                    "rule_id": "rule:mir_handling",
                    "type": "MIR_CAPTURE",
                    "severity": "mandatory",
                    "action": "capture_administrative_only",
                    "message": "Unsolicited Medical Information Request (MIR) detected. Route administratively to Medical Affairs without summarizing clinical content."
                })
            
        return results

    def evaluate_role_scope(self, role: str, detected_topics: List[str], text: str) -> Dict[str, Any]:
        """
        Evaluates whether the candidate answer violates any role-specific scope
        or exclusion rules in the Knowledge Graph.
        """
        role_upper = role.upper()
        out_topics = self.out_of_scope_topics.get(role_upper, set())
        
        violations = []
        for topic in detected_topics:
            if topic in out_topics:
                # Find matching graph exclusion rule
                matching_rule = self._find_exclusion_rule_for_topic(role_upper, topic)
                violations.append({
                    "topic": topic,
                    "rule_id": matching_rule.get("id", "resp:" + role_upper + ":scope"),
                    "rule_text": matching_rule.get("properties", {}).get("text", f"Discussions on {topic} are outside the {role_upper} commercial scope.")
                })
        
        # Specific FRM exclusion checks for clinical keywords even if not full topic
        if role_upper == "FRM":
            clinical_patterns = [
                (r"\b(biomarker|egfr|exon\s*20|ngs)\b", "biomarker testing", "resp:FRM:28", "Ignore all facts related to biomarker testing, disease-state education, and therapeutic decision-making."),
                (r"\b(clinical\s+trial|pfs|overall\s+survival|efficacy)\b", "efficacy safety product info", "resp:FRM:28", "Ignore all facts related to clinical treatment preferences, product efficacy, or therapeutic decision-making."),
                (r"\b(dosing|dose|subcutaneous|infusion\s+protocol)\b", "dosing administration", "resp:FRM:24", "Excluding dosing and administration protocol discussions from FRM summary.")
            ]
            for pat, top_name, rid, rtext in clinical_patterns:
                if re.search(pat, text, re.IGNORECASE) and not any(v["topic"] == top_name for v in violations):
                    violations.append({
                        "topic": top_name,
                        "rule_id": rid,
                        "rule_text": rtext
                    })
        
        is_in_scope = len(violations) == 0
        return {
            "role": role_upper,
            "is_in_scope": is_in_scope,
            "violations": violations,
            "in_scope_topics": list(self.in_scope_topics.get(role_upper, set())),
            "out_of_scope_topics": list(out_topics)
        }

    def _find_exclusion_rule_for_topic(self, role: str, topic: str) -> Dict[str, Any]:
        rules = self.exclusion_rules.get(role, [])
        topic_lower = topic.lower()
        for r in rules:
            text = r.get("properties", {}).get("text", "").lower()
            if topic_lower in text:
                return r
        if rules:
            return rules[0]
        return {"id": f"resp:{role}:scope", "properties": {"text": f"{topic} is out of scope for {role}."}}
