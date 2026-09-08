#!/usr/bin/env python3
"""
generate_role_kgs.py
Generates two separate Knowledge Graph specifications:
- kg_os.json: Scoped to Oncology Specialist (OS) commercial promotional scope,
  including the 5 major accounts and their Patient Identification Barriers.
- kg_frm.json: Scoped to Field Reimbursement Manager (FRM) market access scope,
  including the 5 major accounts and their Market Access Barriers.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_KG_PATH = os.path.join(BASE_DIR, "Persona_Solid_Cancer_OS_FRM.json")

ACCOUNTS = [
    {
        "id": "account:apollo_hospitals",
        "name": "Apollo Hospitals",
        "canonical_name": "Apollo Hospitals",
        "city": "Chennai / Multi-city",
        "type": "Private Healthcare Network",
        "oncology_specialties": ["Urologic Oncology", "Thoracic Oncology", "Medical Oncology"]
    },
    {
        "id": "account:fortis_healthcare",
        "name": "Fortis Healthcare",
        "canonical_name": "Fortis Healthcare",
        "city": "Gurugram / Multi-city",
        "type": "Tertiary Hospital Network",
        "oncology_specialties": ["Surgical Oncology", "Precision Medicine", "Thoracic Oncology"]
    },
    {
        "id": "account:manipal_hospitals",
        "name": "Manipal Hospitals",
        "canonical_name": "Manipal Hospitals",
        "city": "Bengaluru / Multi-city",
        "type": "Comprehensive Cancer Center",
        "oncology_specialties": ["Urologic Oncology", "Medical Oncology", "Infusion Services"]
    },
    {
        "id": "account:max_healthcare",
        "name": "Max Healthcare",
        "canonical_name": "Max Healthcare",
        "city": "Delhi NCR / Multi-city",
        "type": "Quaternary Care Hospital System",
        "oncology_specialties": ["Thoracic Oncology", "Urology & Kidney Transplant", "Pathology Lab"]
    },
    {
        "id": "account:narayana_health",
        "name": "Narayana Health",
        "canonical_name": "Narayana Health",
        "city": "Bengaluru / Multi-city",
        "type": "Integrated Health Network",
        "oncology_specialties": ["Medical Oncology", "Radiation Oncology", "Urologic Oncology"]
    }
]

BARRIERS_OS = [
    {
        "account": "Apollo Hospitals",
        "account_id": "account:apollo_hospitals",
        "barrier_id": "barrier:os:apollo_hospitals",
        "role": "OS",
        "barrier_type": "Patient Identification Barrier",
        "barrier_details": "They are seeing very few eligible patients right now, so the team is not confident they can operationalize the pathway consistently. They asked for a simple way to recognize and flag potential candidates as they appear.",
        "trigger_keywords": [
            "few eligible patients", "eligible patients", "operationalize", "pathway consistently",
            "flag potential candidates", "flag candidates", "recognize candidates", "patient identification",
            "few candidates", "low patient volume", "operationalize the pathway", "pathway friction"
        ],
        "account_ask": "A simple way to recognize and flag potential candidates as they appear.",
        "case_1_inquiry": "Given the difficulty operationalizing the pathway at Apollo Hospitals, did you and {hcp} discuss implementing a streamlined criteria or flagging tool to identify candidates as they appear?",
        "case_2_inquiry": "Regarding account pathway operations at Apollo Hospitals: historical feedback noted seeing few eligible patients and needing a simple way to operationalize the pathway. Did {hcp} or the team mention any friction identifying or flagging eligible candidates today?"
    },
    {
        "account": "Fortis Healthcare",
        "account_id": "account:fortis_healthcare",
        "barrier_id": "barrier:os:fortis_healthcare",
        "role": "OS",
        "barrier_type": "Patient Identification Barrier",
        "barrier_details": "Biomarker testing and/or results are sometimes delayed, which slows down patient identification in their pathway. The team asked to align internally on where testing fits in their workflow and who owns tracking results.",
        "trigger_keywords": [
            "biomarker", "biomarker testing", "results delayed", "delayed testing",
            "slows down patient identification", "align internally", "workflow testing",
            "who owns tracking", "tracking results", "testing turnaround", "molecular results"
        ],
        "account_ask": "Align internally on where testing fits in their workflow and who owns tracking results.",
        "case_1_inquiry": "Regarding the biomarker testing delays at Fortis Healthcare, did you and the team align internally on where testing fits in their clinical workflow and who owns tracking the results?",
        "case_2_inquiry": "In terms of clinical workflow, Fortis Healthcare previously experienced biomarker testing turnaround delays. Did {hcp} mention whether testing delays are currently impacting patient identification, or who tracks the results?"
    },
    {
        "account": "Manipal Hospitals",
        "account_id": "account:manipal_hospitals",
        "barrier_id": "barrier:os:manipal_hospitals",
        "role": "OS",
        "barrier_type": "Patient Identification Barrier",
        "barrier_details": "Key steps in their diagnosis-to-treatment pathway are taking longer than expected due to scheduling and coordination, which delays next-step decisions. The team wants clearer handoffs so potential candidates are identified and routed sooner.",
        "trigger_keywords": [
            "scheduling and coordination", "diagnosis-to-treatment", "taking longer",
            "delays next-step decisions", "clearer handoffs", "routed sooner", "scheduling delays",
            "coordination delays", "handoff delays", "candidate routing"
        ],
        "account_ask": "Clearer handoffs so potential candidates are identified and routed sooner.",
        "case_1_inquiry": "With diagnosis-to-treatment scheduling delays at Manipal Hospitals, did you and the team establish clearer handoffs so potential candidates can be identified and routed sooner?",
        "case_2_inquiry": "At Manipal Hospitals, scheduling and coordination across the diagnosis-to-treatment pathway have previously delayed next-step decisions. Did {hcp} mention any scheduling friction or need for clearer candidate routing handoffs today?"
    },
    {
        "account": "Max Healthcare",
        "account_id": "account:max_healthcare",
        "barrier_id": "barrier:os:max_healthcare",
        "role": "OS",
        "barrier_type": "Patient Identification Barrier",
        "barrier_details": "There are delays in the diagnostic workup and internal handoffs, so potential candidates are identified late in the process. The team wants a cleaner pathway with fewer missed handoffs between roles.",
        "trigger_keywords": [
            "diagnostic workup", "internal handoffs", "identified late", "cleaner pathway",
            "fewer missed handoffs", "missed handoffs", "between roles", "diagnostic delays",
            "late identification", "workup delays"
        ],
        "account_ask": "A cleaner pathway with fewer missed handoffs between roles.",
        "case_1_inquiry": "Given the diagnostic workup delays at Max Healthcare, did you discuss creating a cleaner pathway with fewer missed handoffs between diagnostic and clinical roles?",
        "case_2_inquiry": "Regarding workflow coordination, Max Healthcare has previously seen diagnostic workup delays and late candidate identification. Did {hcp} mention any internal handoff friction or missed handoffs between roles today?"
    },
    {
        "account": "Narayana Health",
        "account_id": "account:narayana_health",
        "barrier_id": "barrier:os:narayana_health",
        "role": "OS",
        "barrier_type": "Patient Identification Barrier",
        "barrier_details": "The site is not consistently identifying potential candidates in their current workflow. They want a simple way to flag appropriate patients earlier without adding extra burden to clinic staff.",
        "trigger_keywords": [
            "not consistently identifying", "current workflow", "flag appropriate patients earlier",
            "flag earlier", "extra burden", "burden to clinic staff", "clinic staff burden",
            "staff burden", "simpler way to flag", "burdening clinic staff"
        ],
        "account_ask": "A simple way to flag appropriate patients earlier without adding extra burden to clinic staff.",
        "case_1_inquiry": "Given that candidate identification has been inconsistent at Narayana Health, did you explore a simplified way to flag appropriate patients earlier without burdening clinic staff?",
        "case_2_inquiry": "Narayana Health has previously sought an easier way to flag eligible candidates earlier without adding burden to clinic staff. Did patient identification or workflow burden come up during your discussion today?"
    }
]

BARRIERS_FRM = [
    {
        "account": "Apollo Hospitals",
        "account_id": "account:apollo_hospitals",
        "barrier_id": "barrier:frm:apollo_hospitals",
        "role": "FRM",
        "barrier_type": "Market Access Barrier",
        "barrier_details": "PA turnaround times are inconsistent, which makes it hard for the team to plan next steps. They asked to use approved access-support resources to understand the process and reduce rework.",
        "trigger_keywords": [
            "pa turnaround", "turnaround times", "inconsistent pa", "prior authorization turnaround",
            "access-support resources", "reduce rework", "rework", "pa delays", "pa timeline"
        ],
        "account_ask": "Use approved access-support resources to understand the process and reduce rework.",
        "case_1_inquiry": "Given the inconsistent PA turnaround times at Apollo Hospitals, did you review approved access-support resources with the team to clarify payer requirements and reduce rework?",
        "case_2_inquiry": "Before we finalize, Apollo Hospitals previously faced inconsistent PA turnaround times. Did the team raise any prior authorization turnaround delays or ask to use approved access-support resources today?"
    },
    {
        "account": "Fortis Healthcare",
        "account_id": "account:fortis_healthcare",
        "barrier_id": "barrier:frm:fortis_healthcare",
        "role": "FRM",
        "barrier_type": "Market Access Barrier",
        "barrier_details": "Patient cost exposure is a concern and the team wants clearer expectations before moving forward. They asked about benefits verification and approved resources that can help set expectations appropriately.",
        "trigger_keywords": [
            "cost exposure", "patient cost", "copay", "co-pay", "benefits verification",
            "financial concern", "clearer expectations", "approved resources", "affordability",
            "out-of-pocket", "out of pocket"
        ],
        "account_ask": "Benefits verification and approved resources that can help set expectations appropriately.",
        "case_1_inquiry": "Regarding the patient cost exposure concerns at Fortis Healthcare, did you discuss benefits verification steps and approved affordability resources to establish clear expectations?",
        "case_2_inquiry": "A key operational consideration at Fortis Healthcare has been patient cost exposure. Did benefits verification, copay support, or patient cost expectations come up during your meeting?"
    },
    {
        "account": "Manipal Hospitals",
        "account_id": "account:manipal_hospitals",
        "barrier_id": "barrier:frm:manipal_hospitals",
        "role": "FRM",
        "barrier_type": "Market Access Barrier",
        "barrier_details": "Prior authorization timelines are creating scheduling uncertainty. The account asked what information their payer typically requests and wanted support using the appropriate, approved access pathway to reduce back-and-forth.",
        "trigger_keywords": [
            "prior authorization timelines", "pa timelines", "scheduling uncertainty",
            "payer typically requests", "approved access pathway", "reduce back-and-forth",
            "back-and-forth", "payer requests", "pa uncertainty", "scheduling delay with pa"
        ],
        "account_ask": "What information their payer typically requests and support using the appropriate, approved access pathway to reduce back-and-forth.",
        "case_1_inquiry": "Regarding the PA scheduling uncertainty at Manipal Hospitals, did you clarify typical payer documentation requests and guide the team on using the approved access pathway to reduce back-and-forth?",
        "case_2_inquiry": "Prior authorization timelines have created scheduling uncertainty at Manipal Hospitals in the past. Did the account ask about typical payer clinical requirements or need access pathway support to reduce back-and-forth?"
    },
    {
        "account": "Max Healthcare",
        "account_id": "account:max_healthcare",
        "barrier_id": "barrier:frm:max_healthcare",
        "role": "FRM",
        "barrier_type": "Market Access Barrier",
        "barrier_details": "The product is not on the current formulary (or is restricted), so usage requires an exception pathway. The account needs clarity on committee timing, required documentation, and who will submit the request.",
        "trigger_keywords": [
            "not on the current formulary", "not on formulary", "restricted",
            "exception pathway", "committee timing", "required documentation",
            "submit the request", "p&t", "formulary restriction", "formulary exception"
        ],
        "account_ask": "Clarity on committee timing, required documentation, and who will submit the request.",
        "case_1_inquiry": "Since the product requires a formulary exception at Max Healthcare, did you clarify P&T committee timing, the required clinical documentation package, and who will submit the request?",
        "case_2_inquiry": "Regarding institutional access at Max Healthcare, since the brand requires a formulary exception pathway, did you discuss committee timing, required documentation, or who will submit the request?"
    },
    {
        "account": "Narayana Health",
        "account_id": "account:narayana_health",
        "barrier_id": "barrier:frm:narayana_health",
        "role": "FRM",
        "barrier_type": "Market Access Barrier",
        "barrier_details": "Recent changes in payer coverage policy created uncertainty about requirements and next steps. The account requested the latest available, published policy information and clarity on how to route questions through the approved access support channel.",
        "trigger_keywords": [
            "recent changes in payer coverage", "coverage policy", "payer coverage policy",
            "policy uncertainty", "published policy", "access support channel", "route questions",
            "payer policy updates", "policy requirements"
        ],
        "account_ask": "The latest available, published policy information and clarity on how to route questions through the approved access support channel.",
        "case_1_inquiry": "Regarding the recent payer policy changes impacting Narayana Health, did you share the latest published coverage policy guidelines and explain how to route questions through the approved access support channel?",
        "case_2_inquiry": "Recent payer coverage policy changes have created uncertainty at Narayana Health. Did the team request updated published policy guidelines or ask how to route questions through approved access support channels?"
    }
]

def build_role_kg(role: str):
    with open(SRC_KG_PATH, "r", encoding="utf-8") as f:
        src = json.load(f)

    target_domain = "commercial_promotional" if role == "OS" else "patient_access_reimbursement"
    barriers = BARRIERS_OS if role == "OS" else BARRIERS_FRM

    # Filter base nodes
    nodes = []
    edges = []
    
    # 1. Add Role node
    role_node = next((n for n in src["nodes"] if n["id"] == f"role:{role}"), None)
    if role_node:
        nodes.append(role_node)

    # 2. Add Domain node
    domain_node = next((n for n in src["nodes"] if n["id"] == f"domain:{target_domain}"), None)
    if domain_node:
        nodes.append(domain_node)

    # 3. Add Topics relevant to this role
    role_edges = [e for e in src["edges"] if e["source"] == f"role:{role}"]
    in_scope_topic_ids = [e["target"] for e in role_edges if e["type"] == "HAS_IN_SCOPE_TOPIC"]
    out_of_scope_topic_ids = [e["target"] for e in role_edges if e["type"] == "HAS_OUT_OF_SCOPE_TOPIC"]
    all_role_topic_ids = set(in_scope_topic_ids + out_of_scope_topic_ids)

    for n in src["nodes"]:
        if n["label"] == "Topic" and n["id"] in all_role_topic_ids:
            nodes.append(n)

    # 4. Add Compliance Rules
    for n in src["nodes"]:
        if n["label"] == "ComplianceRule":
            nodes.append(n)

    # 5. Add CoreResponsibilities & Rules for this role
    for n in src["nodes"]:
        nid = n["id"]
        if (f":{role}:" in nid or nid.endswith(f":{role}")) and n["label"] in [
            "CoreResponsibility", "ExclusionRule", "DocumentationRule", "CollaborationRule"
        ]:
            nodes.append(n)

    # 6. Add Account Nodes
    for acc in ACCOUNTS:
        nodes.append({
            "id": acc["id"],
            "label": "Account",
            "properties": {
                "name": acc["name"],
                "canonical_name": acc["canonical_name"],
                "city": acc["city"],
                "type": acc["type"],
                "oncology_specialties": acc["oncology_specialties"]
            }
        })

    # 7. Add Account Barrier Nodes
    for b in barriers:
        nodes.append({
            "id": b["barrier_id"],
            "label": "AccountBarrier",
            "properties": {
                "account": b["account"],
                "account_id": b["account_id"],
                "role": b["role"],
                "barrier_type": b["barrier_type"],
                "barrier_details": b["barrier_details"],
                "trigger_keywords": b["trigger_keywords"],
                "account_ask": b["account_ask"],
                "case_1_inquiry": b["case_1_inquiry"],
                "case_2_inquiry": b["case_2_inquiry"]
            }
        })

    # 8. Add Existing Edges scoped to this role
    node_ids = set(n["id"] for n in nodes)
    for e in src["edges"]:
        if e["source"] in node_ids and e["target"] in node_ids:
            edges.append(e)

    # 9. Add Edges connecting Role, Accounts, Barriers, and Topics
    for acc in ACCOUNTS:
        edges.append({
            "source": f"role:{role}",
            "target": acc["id"],
            "type": "ENGAGES_ACCOUNT",
            "properties": {"role": role, "account": acc["name"]}
        })

    for b in barriers:
        edges.append({
            "source": b["account_id"],
            "target": b["barrier_id"],
            "type": "HAS_ACCOUNT_BARRIER",
            "properties": {
                "barrier_type": b["barrier_type"],
                "role": b["role"]
            }
        })
        edges.append({
            "source": b["barrier_id"],
            "target": f"role:{role}",
            "type": "GOVERNS_ROLE",
            "properties": {
                "barrier_type": b["barrier_type"]
            }
        })
        primary_topic = "topic:treatment_sequencing" if role == "OS" else "topic:prior_authorization"
        if primary_topic in node_ids:
            edges.append({
                "source": b["barrier_id"],
                "target": primary_topic,
                "type": "RELATES_TO_TOPIC",
                "properties": {
                    "topic": primary_topic.replace("topic:", "")
                }
            })

    output_kg = {
        "schema_version": "3.0",
        "description": f"Dedicated Knowledge Graph for Johnson & Johnson {role} Commercial Oncology Operations with Embedded Account Barrier Intelligence.",
        "role": role,
        "primary_domain": target_domain,
        "node_labels": [
            "Role", "Domain", "Topic", "CoreResponsibility",
            "ExclusionRule", "DocumentationRule", "CollaborationRule",
            "ComplianceRule", "Account", "AccountBarrier"
        ],
        "edge_types": list(set(e["type"] for e in edges)),
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "accounts_indexed": len(ACCOUNTS),
            "barriers_indexed": len(barriers),
            "responsibilities": len([n for n in nodes if n["label"] == "CoreResponsibility"]),
            "rules": len([n for n in nodes if "Rule" in n["label"] and n["label"] != "Role"])
        }
    }

    out_file = os.path.join(BASE_DIR, f"kg_{role.lower()}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_kg, f, indent=2, ensure_ascii=False)
    print(f"Generated {out_file}: {len(nodes)} nodes, {len(edges)} edges.")

if __name__ == "__main__":
    build_role_kg("OS")
    build_role_kg("FRM")
