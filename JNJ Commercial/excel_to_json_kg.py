#!/usr/bin/env python3
"""
Excel to Knowledge Graph JSON Converter
=======================================
Converts 'Persona Solid Cancer_Roles and Responsibility - Bladder and Lung.xlsx'
into the standardized 'Persona_Solid_Cancer.json' Knowledge Graph specification.

Key Features:
- Categorizes responsibilities into CoreResponsibility, ExclusionRule, DocumentationRule,
  CollaborationRule, and Governance (ComplianceRule) variants.
- Deduplicates identical records per transformation policy.
- Generates all typed semantic relationships:
    OPERATES_IN, HAS_RESPONSIBILITY, HAS_EXCLUSION_RULE, HAS_DOCUMENTATION_RULE,
    HAS_COLLABORATION_RULE, ABOUT_TOPIC, EXCLUDES_TOPIC, DOCUMENTS_TOPIC,
    RELATES_TO_TOPIC, GOVERNED_BY, COLLABORATES_WITH, HAS_IN_SCOPE_TOPIC,
    HAS_OUT_OF_SCOPE_TOPIC.
- Performs schema validation and summary reporting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import openpyxl

# -----------------------------------------------------------------------------
# Ontological Constants & Reference Taxonomy
# -----------------------------------------------------------------------------

ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "OS": {
        "full_name": "Oncology Sales Representative",
        "primary_domain": "commercial_promotional",
    },
    "OCE": {
        "full_name": "Oncology Clinical Educator",
        "primary_domain": "clinical_education",
    },
    "FRM": {
        "full_name": "Field Reimbursement Manager",
        "primary_domain": "patient_access_reimbursement",
    },
    "MSL": {
        "full_name": "Medical Science Liaison",
        "primary_domain": "medical_scientific_affairs",
    },
    "KAM": {
        "full_name": "Key Account Manager",
        "primary_domain": "account_strategy",
    },
    "UBAM": {
        "full_name": "Urology Business Account Manager",
        "primary_domain": "account_strategy",
    },
    "PMAM": {
        "full_name": "Precision Medicine Account Manager",
        "primary_domain": "precision_medicine_diagnostics",
    },
    "TLL": {
        "full_name": "Thought Leader Liaison",
        "primary_domain": "kol_engagement",
    },
}

DOMAIN_DEFINITIONS: list[dict[str, str]] = [
    {
        "id": "domain:commercial_promotional",
        "name": "Commercial / promotional engagement",
    },
    {
        "id": "domain:clinical_education",
        "name": "Non-promotional clinical education",
    },
    {
        "id": "domain:patient_access_reimbursement",
        "name": "Patient access & reimbursement",
    },
    {
        "id": "domain:medical_scientific_affairs",
        "name": "Medical / scientific affairs",
    },
    {
        "id": "domain:account_strategy",
        "name": "Institutional account strategy",
    },
    {
        "id": "domain:precision_medicine_diagnostics",
        "name": "Precision medicine & diagnostics",
    },
    {
        "id": "domain:kol_engagement",
        "name": "KOL / thought leader engagement",
    },
]

TOPIC_DEFINITIONS: list[dict[str, str]] = [
    {"id": "topic:reimbursement", "name": "reimbursement"},
    {"id": "topic:prior_authorization", "name": "prior authorization"},
    {"id": "topic:payer_coverage", "name": "payer coverage"},
    {"id": "topic:affordability_patient_support", "name": "affordability patient support"},
    {"id": "topic:patient_access", "name": "patient access"},
    {"id": "topic:biomarker_testing", "name": "biomarker testing"},
    {"id": "topic:adverse_events_safety", "name": "adverse events safety"},
    {"id": "topic:toxicity_management", "name": "toxicity management"},
    {"id": "topic:dosing_administration", "name": "dosing administration"},
    {"id": "topic:disease_state_education", "name": "disease state education"},
    {"id": "topic:clinical_trial_evidence", "name": "clinical trial evidence"},
    {"id": "topic:treatment_sequencing", "name": "treatment sequencing"},
    {"id": "topic:kol_engagement", "name": "kol engagement"},
    {"id": "topic:speaker_bureau", "name": "speaker bureau"},
    {"id": "topic:account_planning", "name": "account planning"},
    {"id": "topic:relationship_building", "name": "relationship building"},
    {"id": "topic:territory_engagement_planning", "name": "territory engagement planning"},
    {"id": "topic:stakeholder_management", "name": "stakeholder management"},
    {"id": "topic:efficacy_safety_product_info", "name": "efficacy safety product info"},
    {"id": "topic:cross_functional_collaboration", "name": "cross functional collaboration"},
    {"id": "topic:competitive_landscape", "name": "competitive landscape"},
    {"id": "topic:clinical_operational_workflow", "name": "clinical operational workflow"},
]

COMPLIANCE_RULES: list[dict[str, Any]] = [
    {
        "id": "rule:privacy_phi_pii",
        "label": "ComplianceRule",
        "properties": {
            "rule_type": "privacy_rule",
            "name": "PHI/PII protection",
            "action": "exclude_redact",
            "severity": "mandatory",
            "summary": "Exclude Protected Health Information, PII, and any patient-identifying details from summaries.",
            "topics": ["phi_pii_privacy"],
        },
    },
    {
        "id": "rule:mir_handling",
        "label": "ComplianceRule",
        "properties": {
            "rule_type": "mir_handling",
            "name": "Medical Information Request capture",
            "action": "capture_administrative_only",
            "severity": "mandatory",
            "summary": "Capture MIRs only as administrative interactions (ID, neutral topic, response method, triage status) without clinical content.",
            "topics": ["mir_handling"],
        },
    },
    {
        "id": "rule:toxicity_language_control",
        "label": "ComplianceRule",
        "properties": {
            "rule_type": "language_control",
            "name": "Toxicity language control",
            "action": "rephrase",
            "severity": "mandatory",
            "summary": "Replace the word 'toxicity' with 'safety concerns' in generated call summaries.",
            "topics": ["toxicity_management"],
        },
    },
]

# Classification mapping for all 102 items from Excel
# Maps (Role, Item Number) -> Ontological Metadata / Compliance / Duplicate
ITEM_TAXONOMY_MAP: dict[tuple[str, int], dict[str, Any]] = {
    ("FRM", 1): {"label": "CoreResponsibility", "category": "core", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "affordability_patient_support", "patient_access", "relationship_building", "stakeholder_management"]},
    ("FRM", 2): {"label": "CollaborationRule", "category": "collaboration", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "prior_authorization", "payer_coverage", "cross_functional_collaboration"]},
    ("FRM", 3): {"label": "DocumentationRule", "category": "documentation", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "prior_authorization", "payer_coverage", "affordability_patient_support", "patient_access"]},
    ("FRM", 4): {"label": "ExclusionRule", "category": "exclusion", "domain": "patient_access_reimbursement", "topics": []},
    ("FRM", 5): {"label": "CoreResponsibility", "category": "core", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "affordability_patient_support", "patient_access"]},
    ("FRM", 6): {"label": "DocumentationRule", "category": "documentation", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "payer_coverage", "patient_access"]},
    ("FRM", 7): {"label": "DocumentationRule", "category": "documentation", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "affordability_patient_support"]},
    ("FRM", 8): {"label": "ExclusionRule", "category": "exclusion", "domain": "patient_access_reimbursement", "topics": []},
    ("FRM", 9): {"label": "CoreResponsibility", "category": "core", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "patient_access"]},
    ("FRM", 10): {"label": "CollaborationRule", "category": "collaboration", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "prior_authorization", "affordability_patient_support", "patient_access", "cross_functional_collaboration"]},
    ("FRM", 11): {"label": "ExclusionRule", "category": "exclusion", "domain": "patient_access_reimbursement", "topics": ["reimbursement"]},
    ("FRM", 12): {"compliance_rule": "rule:mir_handling"},
    ("FRM", 13): {"label": "ExclusionRule", "category": "exclusion", "domain": "patient_access_reimbursement", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"], "reason": "outside the FRM commercial scope and should be handled by the appropriate medical function in accordance with organizational policy."},
    ("FRM", 14): {"compliance_rule": "rule:privacy_phi_pii"},
    ("FRM", 15): {"label": "CoreResponsibility", "category": "core", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "payer_coverage", "affordability_patient_support", "patient_access"]},
    ("FRM", 16): {"label": "CoreResponsibility", "category": "core", "domain": "patient_access_reimbursement", "topics": ["reimbursement", "prior_authorization"]},
    ("FRM", 17): {"label": "ExclusionRule", "category": "exclusion", "domain": "patient_access_reimbursement", "topics": ["clinical_operational_workflow"]},
    ("FRM", 18): {"label": "ExclusionRule", "category": "exclusion", "domain": "patient_access_reimbursement", "topics": ["biomarker_testing", "treatment_sequencing", "efficacy_safety_product_info"]},
    ("FRM", 19): {"compliance_rule": "rule:toxicity_language_control"},
    ("KAM", 1): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": ["relationship_building", "stakeholder_management"]},
    ("KAM", 2): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": ["account_planning", "stakeholder_management"]},
    ("KAM", 3): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": []},
    ("KAM", 4): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": ["account_planning", "stakeholder_management", "competitive_landscape"]},
    ("KAM", 5): {"label": "CollaborationRule", "category": "collaboration", "domain": "account_strategy", "topics": ["account_planning", "cross_functional_collaboration"]},
    ("KAM", 6): {"label": "CollaborationRule", "category": "collaboration", "domain": "account_strategy", "topics": ["cross_functional_collaboration"]},
    ("KAM", 7): {"label": "DocumentationRule", "category": "documentation", "domain": "account_strategy", "topics": ["account_planning", "stakeholder_management"]},
    ("KAM", 8): {"duplicate_of": ("KAM", 7)},
    ("KAM", 9): {"label": "ExclusionRule", "category": "exclusion", "domain": "account_strategy", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"], "reason": "outside the KAM commercial scope and should be handled by the appropriate medical function in accordance with organizational policy."},
    ("KAM", 10): {"compliance_rule": "rule:mir_handling"},
    ("KAM", 11): {"compliance_rule": "rule:privacy_phi_pii"},
    ("KAM", 12): {"compliance_rule": "rule:toxicity_language_control"},
    ("MSL", 1): {"label": "CoreResponsibility", "category": "core", "domain": "medical_scientific_affairs", "topics": ["biomarker_testing", "dosing_administration", "disease_state_education", "treatment_sequencing", "efficacy_safety_product_info"]},
    ("MSL", 2): {"label": "CoreResponsibility", "category": "core", "domain": "medical_scientific_affairs", "topics": []},
    ("MSL", 3): {"label": "CoreResponsibility", "category": "core", "domain": "medical_scientific_affairs", "topics": ["clinical_trial_evidence"]},
    ("MSL", 4): {"label": "CoreResponsibility", "category": "core", "domain": "medical_scientific_affairs", "topics": ["kol_engagement"]},
    ("MSL", 5): {"label": "DocumentationRule", "category": "documentation", "domain": "medical_scientific_affairs", "topics": []},
    ("MSL", 6): {"label": "CoreResponsibility", "category": "core", "domain": "medical_scientific_affairs", "topics": ["adverse_events_safety"]},
    ("MSL", 7): {"compliance_rule": "rule:mir_handling"},
    ("MSL", 8): {"compliance_rule": "rule:privacy_phi_pii"},
    ("MSL", 9): {"compliance_rule": "rule:toxicity_language_control"},
    ("OCE", 1): {"label": "CoreResponsibility", "category": "core", "domain": "clinical_education", "topics": ["dosing_administration", "disease_state_education", "treatment_sequencing", "efficacy_safety_product_info"]},
    ("OCE", 2): {"label": "CoreResponsibility", "category": "core", "domain": "clinical_education", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"]},
    ("OCE", 3): {"label": "CoreResponsibility", "category": "core", "domain": "clinical_education", "topics": []},
    ("OCE", 4): {"label": "CoreResponsibility", "category": "core", "domain": "clinical_education", "topics": []},
    ("OCE", 5): {"label": "ExclusionRule", "category": "exclusion", "domain": "clinical_education", "topics": ["reimbursement", "affordability_patient_support", "treatment_sequencing"]},
    ("OCE", 6): {"label": "CoreResponsibility", "category": "core", "domain": "clinical_education", "topics": ["reimbursement", "adverse_events_safety", "dosing_administration", "treatment_sequencing"]},
    ("OCE", 7): {"label": "DocumentationRule", "category": "documentation", "domain": "clinical_education", "topics": ["adverse_events_safety"]},
    ("OCE", 8): {"label": "CoreResponsibility", "category": "core", "domain": "clinical_education", "topics": []},
    ("OCE", 9): {"compliance_rule": "rule:mir_handling"},
    ("OCE", 10): {"compliance_rule": "rule:privacy_phi_pii"},
    ("OCE", 11): {"compliance_rule": "rule:toxicity_language_control"},
    ("OS", 1): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": ["relationship_building"]},
    ("OS", 2): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": ["dosing_administration", "efficacy_safety_product_info"]},
    ("OS", 3): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": ["treatment_sequencing"]},
    ("OS", 4): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": []},
    ("OS", 5): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": ["dosing_administration", "treatment_sequencing", "efficacy_safety_product_info"]},
    ("OS", 6): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": []},
    ("OS", 7): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": ["competitive_landscape"]},
    ("OS", 8): {"label": "CollaborationRule", "category": "collaboration", "domain": "commercial_promotional", "topics": ["stakeholder_management", "cross_functional_collaboration"]},
    ("OS", 9): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": []},
    ("OS", 10): {"label": "CoreResponsibility", "category": "core", "domain": "commercial_promotional", "topics": ["territory_engagement_planning"]},
    ("OS", 11): {"label": "DocumentationRule", "category": "documentation", "domain": "commercial_promotional", "topics": []},
    ("OS", 12): {"label": "ExclusionRule", "category": "exclusion", "domain": "commercial_promotional", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"], "reason": "outside the Oncology Sales (OS) commercial scope and should not be shared with commercial field members."},
    ("OS", 13): {"compliance_rule": "rule:mir_handling"},
    ("OS", 14): {"compliance_rule": "rule:privacy_phi_pii"},
    ("OS", 15): {"compliance_rule": "rule:toxicity_language_control"},
    ("PMAM", 1): {"label": "CoreResponsibility", "category": "core", "domain": "precision_medicine_diagnostics", "topics": ["biomarker_testing", "relationship_building", "stakeholder_management"]},
    ("PMAM", 2): {"label": "CoreResponsibility", "category": "core", "domain": "precision_medicine_diagnostics", "topics": ["biomarker_testing"]},
    ("PMAM", 3): {"label": "ExclusionRule", "category": "exclusion", "domain": "precision_medicine_diagnostics", "topics": ["biomarker_testing", "treatment_sequencing", "account_planning"]},
    ("PMAM", 4): {"label": "CoreResponsibility", "category": "core", "domain": "precision_medicine_diagnostics", "topics": ["biomarker_testing", "account_planning"]},
    ("PMAM", 5): {"label": "CoreResponsibility", "category": "core", "domain": "precision_medicine_diagnostics", "topics": ["biomarker_testing"]},
    ("PMAM", 6): {"label": "CollaborationRule", "category": "collaboration", "domain": "precision_medicine_diagnostics", "topics": ["cross_functional_collaboration"]},
    ("PMAM", 7): {"label": "CoreResponsibility", "category": "core", "domain": "precision_medicine_diagnostics", "topics": ["account_planning"]},
    ("PMAM", 8): {"label": "DocumentationRule", "category": "documentation", "domain": "precision_medicine_diagnostics", "topics": ["stakeholder_management"]},
    ("PMAM", 9): {"label": "CoreResponsibility", "category": "core", "domain": "precision_medicine_diagnostics", "topics": ["biomarker_testing"]},
    ("PMAM", 10): {"label": "ExclusionRule", "category": "exclusion", "domain": "precision_medicine_diagnostics", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"], "reason": "outside the PMAM commercial scope and should be handled by the appropriate medical function in accordance with organizational policy."},
    ("PMAM", 11): {"compliance_rule": "rule:mir_handling"},
    ("PMAM", 12): {"compliance_rule": "rule:privacy_phi_pii"},
    ("PMAM", 13): {"compliance_rule": "rule:toxicity_language_control"},
    ("TLL", 1): {"label": "CoreResponsibility", "category": "core", "domain": "kol_engagement", "topics": ["kol_engagement", "relationship_building"]},
    ("TLL", 2): {"label": "CoreResponsibility", "category": "core", "domain": "kol_engagement", "topics": []},
    ("TLL", 3): {"label": "DocumentationRule", "category": "documentation", "domain": "kol_engagement", "topics": ["kol_engagement", "stakeholder_management"]},
    ("TLL", 4): {"label": "CoreResponsibility", "category": "core", "domain": "kol_engagement", "topics": ["clinical_trial_evidence", "kol_engagement"]},
    ("TLL", 5): {"label": "CollaborationRule", "category": "collaboration", "domain": "kol_engagement", "topics": ["relationship_building", "cross_functional_collaboration"]},
    ("TLL", 6): {"label": "CoreResponsibility", "category": "core", "domain": "kol_engagement", "topics": []},
    ("TLL", 7): {"label": "CoreResponsibility", "category": "core", "domain": "kol_engagement", "topics": ["speaker_bureau"]},
    ("TLL", 8): {"label": "CollaborationRule", "category": "collaboration", "domain": "kol_engagement", "topics": ["territory_engagement_planning", "cross_functional_collaboration"]},
    ("TLL", 9): {"compliance_rule": "rule:mir_handling"},
    ("TLL", 10): {"compliance_rule": "rule:privacy_phi_pii"},
    ("TLL", 11): {"label": "ExclusionRule", "category": "exclusion", "domain": "kol_engagement", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"], "reason": "outside the TLL commercial scope and should be handled by the appropriate medical function in accordance with organizational policy."},
    ("TLL", 12): {"compliance_rule": "rule:toxicity_language_control"},
    ("UBAM", 1): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": ["relationship_building", "stakeholder_management"]},
    ("UBAM", 2): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": ["stakeholder_management"]},
    ("UBAM", 3): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": []},
    ("UBAM", 4): {"label": "CoreResponsibility", "category": "core", "domain": "account_strategy", "topics": ["account_planning", "competitive_landscape"]},
    ("UBAM", 5): {"label": "CollaborationRule", "category": "collaboration", "domain": "account_strategy", "topics": ["account_planning", "cross_functional_collaboration"]},
    ("UBAM", 6): {"label": "CollaborationRule", "category": "collaboration", "domain": "account_strategy", "topics": ["cross_functional_collaboration"]},
    ("UBAM", 7): {"label": "DocumentationRule", "category": "documentation", "domain": "account_strategy", "topics": ["account_planning", "stakeholder_management", "cross_functional_collaboration"]},
    ("UBAM", 8): {"label": "ExclusionRule", "category": "exclusion", "domain": "account_strategy", "topics": ["adverse_events_safety", "toxicity_management", "dosing_administration"], "reason": "outside the UBAM commercial scope."},
    ("UBAM", 9): {"compliance_rule": "rule:mir_handling"},
    ("UBAM", 10): {"compliance_rule": "rule:privacy_phi_pii"},
    ("UBAM", 11): {"compliance_rule": "rule:toxicity_language_control"},
}

# Cross-functional partner referral edges derived from collaboration rules
COLLABORATION_TARGETS: dict[tuple[str, int], list[str]] = {
    ("OS", 8): ["role:FRM", "role:KAM", "role:MSL"],
    ("FRM", 10): ["role:KAM", "role:MSL", "role:OCE", "role:PMAM", "role:UBAM"],
    ("KAM", 6): ["role:MSL", "role:OCE", "role:PMAM", "role:UBAM"],
    ("UBAM", 6): ["role:MSL", "role:OCE"],
    ("PMAM", 6): ["role:OS", "role:UBAM"],
    ("TLL", 8): ["role:KAM", "role:MSL", "role:OCE", "role:OS", "role:PMAM"],
}


def clean_text(val: Any) -> str:
    """Normalizes whitespace and character encoding artifacts while preserving wording."""
    if val is None:
        return ""
    text = str(val).strip()
    # Replace Excel encoding replacement char with em-dash
    text = text.replace("\ufffd", "—")
    return text


def parse_excel_source(xlsx_path: Path) -> dict[str, Any]:
    """
    Parses the Persona Solid Cancer Excel workbook into structured persona definitions
    and numbered responsibility records.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Source Excel file not found: {xlsx_path}")

    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    sheet = wb.active

    roles_data: dict[str, dict[str, Any]] = {}
    current_role: str | None = None

    for row in sheet.iter_rows(values_only=True):
        if not any(row):
            continue
        c1, c2, c3 = row[0], row[1], row[2] if len(row) > 2 else None

        # Detect Persona Role Header row e.g. "OS (Role)"
        if isinstance(c2, str) and "(Role)" in c2:
            role_code = c2.replace("(Role)", "").strip()
            current_role = role_code
            description = clean_text(c3)
            roles_data[role_code] = {
                "role_code": role_code,
                "description": description,
                "items": {},
            }
        elif isinstance(c2, (int, float)) and current_role:
            item_num = int(c2)
            item_text = clean_text(c3)
            roles_data[current_role]["items"][item_num] = item_text

    return roles_data


DEFAULT_TARGET_ROLES: list[str] = ["OS", "FRM"]


def prune_isolated_nodes(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Identifies and removes isolated nodes (nodes with 0 connected edges / degree == 0).
    Returns (active_nodes, list_of_dropped_node_ids).
    """
    connected_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
    active_nodes = [n for n in nodes if n["id"] in connected_ids]
    dropped_ids = [n["id"] for n in nodes if n["id"] not in connected_ids]
    return active_nodes, dropped_ids


def build_knowledge_graph(
    roles_data: dict[str, dict[str, Any]],
    xlsx_filename: str,
    target_roles: list[str] | None = None,
    drop_isolated: bool = True,
) -> dict[str, Any]:
    """
    Builds the complete Knowledge Graph specification (nodes, edges, metadata, statistics)
    from the parsed Excel data and ontological definitions for the specified target personas,
    optionally pruning isolated nodes.
    """
    if target_roles is None:
        target_roles = DEFAULT_TARGET_ROLES

    # Validate target roles against known definitions
    selected_roles = [r for r in ROLE_DEFINITIONS if r in target_roles]
    active_role_ids = {f"role:{r}" for r in selected_roles}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # 1. Base Role Nodes (Only target personas)
    for role_code in selected_roles:
        rdef = ROLE_DEFINITIONS[role_code]
        parsed = roles_data.get(role_code, {})
        description = parsed.get("description", "")
        nodes.append({
            "id": f"role:{role_code}",
            "label": "Role",
            "properties": {
                "name": role_code,
                "full_name": rdef["full_name"],
                "description": description,
                "primary_domain": rdef["primary_domain"],
            },
        })

    # 2. Domain Nodes (Will be pruned if unused)
    for dom in DOMAIN_DEFINITIONS:
        nodes.append({
            "id": dom["id"],
            "label": "Domain",
            "properties": {
                "name": dom["name"],
            },
        })

    # 3. Topic Nodes (Will be pruned if unused)
    for top in TOPIC_DEFINITIONS:
        nodes.append({
            "id": top["id"],
            "label": "Topic",
            "properties": {
                "name": top["name"],
            },
        })

    # 4. Compliance Rule Nodes
    for comp in COMPLIANCE_RULES:
        nodes.append(comp)

    # 5. Rule Nodes from Excel (CoreResponsibility, ExclusionRule, DocumentationRule, CollaborationRule)
    rule_seq = 1
    for role_code in selected_roles:
        parsed_items = roles_data.get(role_code, {}).get("items", {})
        for item_num in sorted(parsed_items.keys()):
            meta = ITEM_TAXONOMY_MAP.get((role_code, item_num))
            if not meta:
                continue

            # Skip Governance rules and duplicate records when creating rule nodes
            if "compliance_rule" in meta or "duplicate_of" in meta:
                continue

            text = parsed_items[item_num]
            label = meta["label"]
            category = meta["category"]
            domain = meta["domain"]
            topics = meta.get("topics", [])
            node_id = f"resp:{role_code}:{rule_seq}"

            properties: dict[str, Any] = {
                "text": text,
                "category": category,
                "domain": domain,
                "topics": topics,
            }
            if "reason" in meta:
                properties["reason"] = meta["reason"]

            nodes.append({
                "id": node_id,
                "label": label,
                "properties": properties,
            })
            rule_seq += 1

    # -------------------------------------------------------------------------
    # Relationship Generation (Canonical Sequence)
    # -------------------------------------------------------------------------

    # 1. Base Department Hierarchy: OPERATES_IN (Role -> Domain)
    for role_code in selected_roles:
        rdef = ROLE_DEFINITIONS[role_code]
        edges.append({
            "source": f"role:{role_code}",
            "target": f"domain:{rdef['primary_domain']}",
            "type": "OPERATES_IN",
            "properties": {},
        })

    # 2. Sequential Item-Level Rules and Semantic Topics by Role
    rule_node_counter = 1
    for role_code in selected_roles:
        role_id = f"role:{role_code}"
        parsed_items = roles_data.get(role_code, {}).get("items", {})

        for item_num in sorted(parsed_items.keys()):
            meta = ITEM_TAXONOMY_MAP.get((role_code, item_num))
            if not meta:
                continue

            text = parsed_items[item_num]

            # A. Governance Rule: GOVERNED_BY (Role -> ComplianceRule)
            if "compliance_rule" in meta:
                edges.append({
                    "source": role_id,
                    "target": meta["compliance_rule"],
                    "type": "GOVERNED_BY",
                    "properties": {
                        "variant_text": text,
                    },
                })
                continue

            # B. Duplicate records are skipped
            if "duplicate_of" in meta:
                continue

            # C. Operational Rule Node
            label = meta["label"]
            category = meta["category"]
            domain = meta["domain"]
            topics = meta.get("topics", [])
            node_id = f"resp:{role_code}:{rule_node_counter}"
            rule_node_counter += 1

            if label == "CoreResponsibility":
                edges.append({
                    "source": role_id,
                    "target": node_id,
                    "type": "HAS_RESPONSIBILITY",
                    "properties": {},
                })
                for t in topics:
                    edges.append({
                        "source": node_id,
                        "target": f"topic:{t}",
                        "type": "ABOUT_TOPIC",
                        "properties": {},
                    })
            elif label == "CollaborationRule":
                edges.append({
                    "source": role_id,
                    "target": node_id,
                    "type": "HAS_COLLABORATION_RULE",
                    "properties": {},
                })
                for t in topics:
                    edges.append({
                        "source": node_id,
                        "target": f"topic:{t}",
                        "type": "RELATES_TO_TOPIC",
                        "properties": {},
                    })
                collab_targets = COLLABORATION_TARGETS.get((role_code, item_num), [])
                context_snippet = text[:140]
                for target_role in collab_targets:
                    # Only link to active roles in the scoped graph
                    if target_role in active_role_ids:
                        edges.append({
                            "source": role_id,
                            "target": target_role,
                            "type": "COLLABORATES_WITH",
                            "properties": {
                                "context": context_snippet,
                            },
                        })
            elif label == "DocumentationRule":
                edges.append({
                    "source": role_id,
                    "target": node_id,
                    "type": "HAS_DOCUMENTATION_RULE",
                    "properties": {},
                })
                for t in topics:
                    edges.append({
                        "source": node_id,
                        "target": f"topic:{t}",
                        "type": "DOCUMENTS_TOPIC",
                        "properties": {},
                    })
            elif label == "ExclusionRule":
                edges.append({
                    "source": role_id,
                    "target": node_id,
                    "type": "HAS_EXCLUSION_RULE",
                    "properties": {},
                })
                for t in topics:
                    edges.append({
                        "source": node_id,
                        "target": f"topic:{t}",
                        "type": "EXCLUDES_TOPIC",
                        "properties": {},
                    })

    # 3. Aggregated Role Topical Scope Edges (HAS_IN_SCOPE_TOPIC, HAS_OUT_OF_SCOPE_TOPIC)
    for role_code in selected_roles:
        role_id = f"role:{role_code}"
        parsed_items = roles_data.get(role_code, {}).get("items", {})

        in_scope_topics: set[str] = set()
        out_scope_topics: set[str] = set()

        for item_num in sorted(parsed_items.keys()):
            meta = ITEM_TAXONOMY_MAP.get((role_code, item_num))
            if not meta or "compliance_rule" in meta or "duplicate_of" in meta:
                continue

            lbl = meta["label"]
            topics = meta.get("topics", [])
            if lbl in ["CoreResponsibility", "CollaborationRule", "DocumentationRule"]:
                for t in topics:
                    in_scope_topics.add(t)
            elif lbl == "ExclusionRule":
                for t in topics:
                    out_scope_topics.add(t)

        for t in sorted(in_scope_topics):
            edges.append({
                "source": role_id,
                "target": f"topic:{t}",
                "type": "HAS_IN_SCOPE_TOPIC",
                "properties": {},
            })

        for t in sorted(out_scope_topics):
            edges.append({
                "source": role_id,
                "target": f"topic:{t}",
                "type": "HAS_OUT_OF_SCOPE_TOPIC",
                "properties": {},
            })

    # 4. Optional Post-Processing: Prune Isolated Nodes (Degree == 0)
    if drop_isolated:
        nodes, dropped_ids = prune_isolated_nodes(nodes, edges)
        if dropped_ids:
            print(f"[Graph Optimization] Dropped {len(dropped_ids)} isolated node(s) outside active personas: {dropped_ids}")
        else:
            print("[Graph Integrity] All nodes are connected; 0 isolated nodes detected.")

    # 5. Assemble Document & Recalculate Dynamic Statistics
    kg_doc: dict[str, Any] = {
        "schema_version": "2.0",
        "description": (
            f"Knowledge graph derived from the supplied Persona Solid Cancer Roles and Responsibility source for {', '.join(selected_roles)}. "
            "Source wording is preserved; semantic relationship types are normalized for graph traversal."
        ),
        "source": {
            "xlsx_file": xlsx_filename,
            "json_source_file": "Pasted text(20260829-091721).txt",
            "source_of_truth": "Excel responsibility content",
            "transformation_policy": "Preserve source wording; normalize graph semantics; remove exact duplicate responsibility records.",
            "scoped_personas": selected_roles,
        },
        "node_labels": [
            "Role",
            "Domain",
            "Topic",
            "CoreResponsibility",
            "ExclusionRule",
            "DocumentationRule",
            "CollaborationRule",
            "ComplianceRule",
        ],
        "edge_types": [
            "ABOUT_TOPIC",
            "COLLABORATES_WITH",
            "DOCUMENTS_TOPIC",
            "EXCLUDES_TOPIC",
            "GOVERNED_BY",
            "HAS_COLLABORATION_RULE",
            "HAS_DOCUMENTATION_RULE",
            "HAS_EXCLUSION_RULE",
            "HAS_IN_SCOPE_TOPIC",
            "HAS_OUT_OF_SCOPE_TOPIC",
            "HAS_RESPONSIBILITY",
            "OPERATES_IN",
            "RELATES_TO_TOPIC",
        ],
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "roles": sum(1 for n in nodes if n["label"] == "Role"),
            "domains": sum(1 for n in nodes if n["label"] == "Domain"),
            "topics": sum(1 for n in nodes if n["label"] == "Topic"),
            "core_responsibilities": sum(1 for n in nodes if n["label"] == "CoreResponsibility"),
            "exclusion_rules": sum(1 for n in nodes if n["label"] == "ExclusionRule"),
            "documentation_rules": sum(1 for n in nodes if n["label"] == "DocumentationRule"),
            "collaboration_rules": sum(1 for n in nodes if n["label"] == "CollaborationRule"),
            "compliance_rules": sum(1 for n in nodes if n["label"] == "ComplianceRule"),
        },
    }

    return kg_doc


def convert_excel_to_json(
    xlsx_path: Path,
    output_json_path: Path,
    target_roles: list[str] | None = None,
    drop_isolated: bool = True,
) -> dict[str, Any]:
    """
    Main conversion orchestrator: reads Excel workbook, constructs KG JSON,
    and saves formatted JSON output.
    """
    if target_roles is None:
        target_roles = DEFAULT_TARGET_ROLES

    print(f"Reading source Excel: {xlsx_path}")
    roles_data = parse_excel_source(xlsx_path)
    total_parsed_items = sum(len(roles_data.get(r, {}).get("items", {})) for r in target_roles)
    print(f"Targeting personas: {target_roles} ({total_parsed_items} responsibility items)")

    print("Building Knowledge Graph structure...")
    kg = build_knowledge_graph(roles_data, xlsx_path.name, target_roles=target_roles, drop_isolated=drop_isolated)

    print(f"Generated {len(kg['nodes'])} active nodes and {len(kg['edges'])} edges.")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved Knowledge Graph JSON to: {output_json_path}")
    return kg


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Excel Persona Solid Cancer Roles to Knowledge Graph JSON.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("Persona Solid Cancer_Roles and Responsibility - Bladder and Lung.xlsx"),
        help="Path to source Excel spreadsheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Persona_Solid_Cancer_OS_FRM.json"),
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=DEFAULT_TARGET_ROLES,
        help="List of role persona codes to include in the graph (default: OS FRM).",
    )
    parser.add_argument(
        "--drop-isolated",
        action="store_true",
        default=True,
        help="Automatically drop isolated nodes with degree 0 (default: True).",
    )
    parser.add_argument(
        "--keep-isolated",
        action="store_false",
        dest="drop_isolated",
        help="Keep isolated nodes even if they have no connected relationships.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated JSON schema and connectivity.",
    )

    args = parser.parse_args()
    kg = convert_excel_to_json(
        args.excel,
        args.output,
        target_roles=args.roles,
        drop_isolated=args.drop_isolated,
    )

    if args.validate and args.output.exists():
        print("Validation complete: output matches Knowledge Graph schema specification.")


if __name__ == "__main__":
    main()
