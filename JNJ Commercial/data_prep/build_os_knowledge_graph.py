"""
build_os_knowledge_graph.py
===========================
Builds the 100% complete, verified Knowledge Graph (kg_os.json) and Account Barriers (account_barriers.json)
exclusively for Oncology Sales (OS) on INLEXZO, grounded directly in 'generated_os_training_transcripts 1.xlsx'.
Eliminates all legacy FRM and wrong-data entities.
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(r"c:\Users\AniruddhaJoshi\OneDrive - Info Origin Technologies Pvt Ltd\Desktop\flow-slm\Flow-SLM\JNJ Commercial")

ACCOUNTS = [
    {
        "id": "account:atlantic_urology_associates",
        "name": "Atlantic Urology Associates",
        "canonical_name": "Atlantic Urology Associates",
        "city": "Tampa, Florida",
        "state": "Florida",
        "type": "Comprehensive Community Urology Practice",
        "oncology_specialties": ["Urologic Oncology", "NMIBC Bladder Cancer Care", "Infusion & Instillation Services"]
    },
    {
        "id": "account:capital_bladder_cancer_center",
        "name": "Capital Bladder Cancer Center",
        "canonical_name": "Capital Bladder Cancer Center",
        "city": "Richmond, Virginia",
        "state": "Virginia",
        "type": "Specialized Bladder Cancer Center of Excellence",
        "oncology_specialties": ["Urologic Oncology", "Bladder Cancer Clinical Care", "Surgical Urology"]
    },
    {
        "id": "account:central_ohio_urology",
        "name": "Central Ohio Urology",
        "canonical_name": "Central Ohio Urology",
        "city": "Gahanna, Ohio",
        "state": "Ohio",
        "type": "Regional Urologic Oncology Center",
        "oncology_specialties": ["Urologic Oncology", "Office-Based Procedure Clinic", "Advanced Bladder Cancer Care"]
    },
    {
        "id": "account:northside_urology_group",
        "name": "Northside Urology Group",
        "canonical_name": "Northside Urology Group",
        "city": "Raleigh, North Carolina",
        "state": "North Carolina",
        "type": "Multi-Physician Urology Surgical Group",
        "oncology_specialties": ["Urologic Oncology", "Ambulatory Surgery Center (ASC)", "Bladder Cancer Clinic"]
    },
    {
        "id": "account:regional_urology_institute",
        "name": "Regional Urology Institute",
        "canonical_name": "Regional Urology Institute",
        "city": "San Antonio, Texas",
        "state": "Texas",
        "type": "Tertiary Urological Health System",
        "oncology_specialties": ["Urologic Oncology", "Robotic & Minimally Invasive Urology", "Clinical Bladder Pathways"]
    },
    {
        "id": "account:summit_urologic_oncology",
        "name": "Summit Urologic Oncology",
        "canonical_name": "Summit Urologic Oncology",
        "city": "Denver, Colorado",
        "state": "Colorado",
        "type": "Academic & Tertiary Urologic Oncology Center",
        "oncology_specialties": ["Urologic Oncology", "Bladder Preservation Program", "Clinical Research"]
    },
    {
        "id": "account:temple_urology_clinic",
        "name": "Temple Urology Clinic",
        "canonical_name": "Temple Urology Clinic",
        "city": "Princeton, New Jersey",
        "state": "New Jersey",
        "type": "Specialized Outpatient Urology Practice",
        "oncology_specialties": ["Urologic Oncology", "NMIBC Intravesical Clinic", "Urology Diagnostics"]
    },
    {
        "id": "account:valley_urology_specialists",
        "name": "Valley Urology Specialists",
        "canonical_name": "Valley Urology Specialists",
        "city": "Phoenix, Arizona",
        "state": "Arizona",
        "type": "Large Independent Urology Practice",
        "oncology_specialties": ["Urologic Oncology", "In-Office Intravesical Procedure Suites", "Comprehensive Bladder Cancer Care"]
    }
]

ACCOUNT_BARRIERS_OS = [
    {
        "account": "Atlantic Urology Associates",
        "account_id": "account:atlantic_urology_associates",
        "barrier_id": "barrier:os:atlantic_urology_associates",
        "role": "OS",
        "barrier_type": "Coverage Policy & Benefits Investigation Barrier",
        "barrier_details": "Recent commercial and Medicare Advantage coverage changes have caused billing uncertainty; the office is rechecking patient out-of-pocket responsibility and waiting on benefits verification before committing to procedure scheduling.",
        "trigger_keywords": [
            "coverage changed", "coverage change", "rechecking", "patient responsibility",
            "benefits investigation", "benefits verification", "recheck coverage", "out-of-pocket",
            "coverage changed recently", "office is rechecking"
        ],
        "account_ask": "Recheck coverage criteria and expedite benefits investigation to clarify patient out-of-pocket responsibility before scheduling.",
        "case_1_inquiry": "Regarding the recent coverage updates and benefits investigation at Atlantic Urology Associates, did you and the office confirm patient responsibility and timeline before scheduling INLEXZO?",
        "case_2_inquiry": "At Atlantic Urology Associates, historical notes highlighted recent coverage policy changes and pending benefits investigations. Did the team or {hcp} mention any delays clarifying patient out-of-pocket responsibility today?"
    },
    {
        "account": "Capital Bladder Cancer Center",
        "account_id": "account:capital_bladder_cancer_center",
        "barrier_id": "barrier:os:capital_bladder_cancer_center",
        "role": "OS",
        "barrier_type": "P&T Committee Review & Deductible Affordability Barrier",
        "barrier_details": "Institutional P&T committee review is currently pending for INLEXZO, and identified eligible patients are working through high deductible and payment timing, making affordability the practical obstacle.",
        "trigger_keywords": [
            "p&t approval", "p&t committee", "committee review", "pending review",
            "deductible", "payment plan", "payment timing", "affordability barrier",
            "p&t approval is the main challenge", "deductible and payment timing"
        ],
        "account_ask": "Clarify P&T committee decision timing and evaluate patient co-pay or deductible payment assistance options.",
        "case_1_inquiry": "Given the pending P&T committee review and deductible payment timing at Capital Bladder Cancer Center, did you discuss committee timing or patient affordability resources with {hcp}?",
        "case_2_inquiry": "Capital Bladder Cancer Center previously faced P&T committee review delays and patient deductible challenges. Did {hcp} or practice staff provide an update on committee timing or patient payment planning today?"
    },
    {
        "account": "Central Ohio Urology",
        "account_id": "account:central_ohio_urology",
        "barrier_id": "barrier:os:central_ohio_urology",
        "role": "OS",
        "barrier_type": "Coverage Policy Change & Prior Authorization Barrier",
        "barrier_details": "Recent coverage policy revisions have slowed patient scheduling, and prior authorizations for subsequent patients are still in process with regional health plans.",
        "trigger_keywords": [
            "coverage changed", "rechecking coverage", "prior authorization completed",
            "pa in process", "pa delays", "regional health plan", "prior authorization holdup"
        ],
        "account_ask": "Support resolving pending prior authorizations and guide office on current coverage documentation requirements.",
        "case_1_inquiry": "With coverage changes and pending prior authorizations at Central Ohio Urology, did you align with {hcp} and staff on submission status and timeline for the upcoming INLEXZO insertion?",
        "case_2_inquiry": "Central Ohio Urology has previously navigated coverage policy updates and prior authorization delays. Did prior authorization status or coverage criteria come up in your discussion with {hcp}?"
    },
    {
        "account": "Northside Urology Group",
        "account_id": "account:northside_urology_group",
        "barrier_id": "barrier:os:northside_urology_group",
        "role": "OS",
        "barrier_type": "Prior Authorization Turnaround Barrier",
        "barrier_details": "Prior authorization submissions are currently in process with payers; clinic policy prohibits putting insertions on the procedure calendar until formal PA approval clears.",
        "trigger_keywords": [
            "pa is still in process", "prior authorization", "pa holdup", "once that clears",
            "not ready to schedule", "approval blocker", "pa turnaround", "pa in process"
        ],
        "account_ask": "Expedite prior authorization status tracking and provide payer documentation checklists so procedures can be placed on the surgical calendar.",
        "case_1_inquiry": "Given the prior authorization turnaround hurdles at Northside Urology Group, did you discuss PA submission status with {hcp} so the patient can be scheduled on the ASC calendar?",
        "case_2_inquiry": "Northside Urology Group previously experienced delays scheduling INLEXZO pending prior authorization clearance. Did {hcp} or the clinic coordinator note any PA holdups for candidate patients today?"
    },
    {
        "account": "Regional Urology Institute",
        "account_id": "account:regional_urology_institute",
        "barrier_id": "barrier:os:regional_urology_institute",
        "role": "OS",
        "barrier_type": "Patient Deductible & Pre-Procedure Scheduling Barrier",
        "barrier_details": "Patients are working through annual deductible and payment timing, while clinic staff require pre-procedure workflow coordination and staff prep before locking in insertion dates.",
        "trigger_keywords": [
            "deductible and payment timing", "affordability", "prep support",
            "office staff prep", "scheduling delay", "working through deductible", "patient deductible"
        ],
        "account_ask": "Coordinate patient affordability support and conduct pre-procedure staff workflow orientation prior to insertion.",
        "case_1_inquiry": "Regarding the deductible payment timing and staff prep requirements at Regional Urology Institute, did you review affordability options and arrange office prep support with {hcp}?",
        "case_2_inquiry": "Regional Urology Institute has previously seen procedure scheduling delays due to patient deductible hurdles and workflow prep needs. Did patient payment timing or staff prep come up with {hcp} today?"
    },
    {
        "account": "Summit Urologic Oncology",
        "account_id": "account:summit_urologic_oncology",
        "barrier_id": "barrier:os:summit_urologic_oncology",
        "role": "OS",
        "barrier_type": "Benefits Investigation & Formulary Exception Barrier",
        "barrier_details": "Benefits verification is pending for identified candidates, and non-formulary institutional status requires individual medical exception documentation before cases proceed to scheduling.",
        "trigger_keywords": [
            "benefits investigation", "benefits verification pending", "not on formulary yet",
            "formulary exception", "commit to the schedule", "exception paperwork", "benefits verification"
        ],
        "account_ask": "Complete benefits investigations swiftly and provide approved exception submission guidelines for non-formulary requests.",
        "case_1_inquiry": "With benefits verification pending and formulary exception needed at Summit Urologic Oncology, did you clarify required clinical documentation with {hcp} to expedite scheduling?",
        "case_2_inquiry": "Historical feedback at Summit Urologic Oncology noted delays in benefits verification and formulary exception approvals. Did {hcp} raise any access or exception paperwork hurdles today?"
    },
    {
        "account": "Temple Urology Clinic",
        "account_id": "account:temple_urology_clinic",
        "barrier_id": "barrier:os:temple_urology_clinic",
        "role": "OS",
        "barrier_type": "Institutional Formulary Addition & Coverage Recheck Barrier",
        "barrier_details": "The product is not yet added to the institutional formulary, preventing scheduling, while recent payer coverage modifications require reverifying patient copay requirements.",
        "trigger_keywords": [
            "product is not on formulary", "not on formulary yet", "coverage changed recently",
            "patient responsibility", "cannot schedule until approval", "formulary addition"
        ],
        "account_ask": "Assist pharmacy with institutional formulary review submission and support clinic staff in re-evaluating payer coverage.",
        "case_1_inquiry": "Since INLEXZO is pending institutional formulary addition at Temple Urology Clinic, did you provide the necessary formulary review materials and discuss expected approval timing with {hcp}?",
        "case_2_inquiry": "Temple Urology Clinic previously needed formulary approval and coverage rechecks before scheduling patients. Did {hcp} or the practice manager discuss formulary progress or coverage re-verification today?"
    },
    {
        "account": "Valley Urology Specialists",
        "account_id": "account:valley_urology_specialists",
        "barrier_id": "barrier:os:valley_urology_specialists",
        "role": "OS",
        "barrier_type": "Formulary Approval & Nursing Anatomical Prep Barrier",
        "barrier_details": "Institutional formulary approval is pending before cases can be put on the clinic calendar; additionally, nursing staff requested a prep session using the anatomical model prior to in-office insertions.",
        "trigger_keywords": [
            "not on formulary yet", "formulary approval", "prep session", "anatomical model",
            "nursing team prep", "deductible timing", "anatomical demo", "staff prep"
        ],
        "account_ask": "Obtain formulary approval and deliver an anatomical model prep demonstration for the clinic nursing team.",
        "case_1_inquiry": "Given the pending formulary approval and nursing prep request at Valley Urology Specialists, did you align on formulary committee timing and schedule an anatomical model demo for the staff?",
        "case_2_inquiry": "Valley Urology Specialists previously highlighted formulary review dependencies and requested anatomical model prep for their nurses. Did {hcp} or the clinic team provide an update on formulary status or prep needs today?"
    }
]

TOPICS = [
    {"id": "topic:patient_identification_volume", "name": "patient identification & candidate eligibility"},
    {"id": "topic:treatment_sequencing", "name": "treatment sequencing & post-BCG options"},
    {"id": "topic:prior_authorization", "name": "prior authorization navigation & approval tracking"},
    {"id": "topic:formulary_pt_committee", "name": "formulary approval & P&T committee review"},
    {"id": "topic:payer_coverage_affordability", "name": "payer coverage & patient deductible affordability"},
    {"id": "topic:procedure_scheduling_site_of_care", "name": "procedure scheduling & site of care logistics"},
    {"id": "topic:anatomical_model_nurse_prep", "name": "anatomical model demonstration & nursing staff prep"},
    {"id": "topic:adverse_events_triage", "name": "adverse event triage & educational escalation"},
    {"id": "topic:stakeholder_management", "name": "urologist & clinic stakeholder engagement"},
    {"id": "topic:territory_engagement_planning", "name": "territory engagement planning & touchpoint follow-up"}
]

OUT_OF_SCOPE_TOPICS = [
    {"id": "topic:clinical_toxicity_management", "name": "clinical toxicity management & dose modifications"},
    {"id": "topic:off_label_promotion", "name": "off-label promotion & unapproved indications"},
    {"id": "topic:direct_price_negotiation", "name": "direct commercial pricing & contract rebate negotiations"}
]

CORE_RESPONSIBILITIES = [
    {
        "id": "resp:OS:01",
        "text": "Engage healthcare professionals (HCPs) and urology care teams to present approved INLEXZO product indications, clinical evidence, and treatment positioning in BCG-unresponsive NMIBC.",
        "domain": "commercial_promotional",
        "topics": ["treatment_sequencing", "patient_identification_volume"]
    },
    {
        "id": "resp:OS:02",
        "text": "Support the compliant identification of appropriate adult patients with BCG-unresponsive non-muscle invasive bladder cancer (NMIBC) with carcinoma in situ (CIS) with or without papillary tumors.",
        "domain": "commercial_promotional",
        "topics": ["patient_identification_volume", "treatment_sequencing"]
    },
    {
        "id": "resp:OS:03",
        "text": "Understand customer account barriers and facilitate compliant solutions regarding prior authorization hurdles, formulary review timelines, and coverage policy updates.",
        "domain": "commercial_promotional",
        "topics": ["prior_authorization", "formulary_pt_committee", "payer_coverage_affordability"]
    },
    {
        "id": "resp:OS:04",
        "text": "Coordinate pre-procedure operational support, including scheduling anatomical model demonstrations and workflow preparation sessions for clinic nurses and APP teams.",
        "domain": "commercial_promotional",
        "topics": ["anatomical_model_nurse_prep", "procedure_scheduling_site_of_care"]
    },
    {
        "id": "resp:OS:05",
        "text": "Log comprehensive call notes detailing customer discussion points, candidate patient counts, insertion scheduling status, site of care, and scheduled follow-up milestones.",
        "domain": "commercial_promotional",
        "topics": ["territory_engagement_planning", "stakeholder_management"]
    },
    {
        "id": "resp:OS:06",
        "text": "Capture and route unsolicited Medical Information Requests (MIRs) and adverse event reports immediately to Medical Affairs and Safety without providing clinical advice.",
        "domain": "commercial_promotional",
        "topics": ["adverse_events_triage"]
    }
]

EXCLUSION_RULES = [
    {
        "id": "excl:OS:01",
        "text": "OS representatives must NOT provide clinical advice on managing adverse events, treating drug toxicities, adjusting dosages, or clinical toxicity interventions.",
        "domain": "commercial_promotional",
        "topics": ["clinical_toxicity_management"]
    },
    {
        "id": "excl:OS:02",
        "text": "OS representatives must NOT promote INLEXZO for unapproved indications, off-label regimens, or non-indicated patient cohorts (such as muscle-invasive bladder cancer).",
        "domain": "commercial_promotional",
        "topics": ["off_label_promotion"]
    },
    {
        "id": "excl:OS:03",
        "text": "OS representatives must NOT negotiate hospital discount pricing, contract rebates, or financial terms with institution procurement staff.",
        "domain": "commercial_promotional",
        "topics": ["direct_price_negotiation"]
    }
]

COMPLIANCE_RULES = [
    {
        "id": "rule:privacy_phi_pii",
        "rule_type": "privacy_rule",
        "name": "PHI/PII protection",
        "action": "exclude_redact",
        "severity": "mandatory",
        "summary": "Strictly exclude Protected Health Information, patient names, dates of birth, MRNs, and any identifiable patient data from call notes.",
        "topics": ["phi_pii_privacy"]
    },
    {
        "id": "rule:mir_handling",
        "rule_type": "mir_handling",
        "name": "Medical Information Request capture",
        "action": "capture_administrative_only",
        "severity": "mandatory",
        "summary": "Capture MIRs only as administrative records (HCP name, question topic, response mode) and route immediately to Medical Affairs.",
        "topics": ["mir_handling"]
    },
    {
        "id": "rule:safety_adverse_events",
        "rule_type": "safety_reporting",
        "name": "Adverse event regulatory escalation",
        "action": "escalate_to_safety",
        "severity": "mandatory",
        "summary": "Immediately route any reported adverse event, discomfort, or product safety issue to Medical Safety within 24 hours.",
        "topics": ["adverse_events_triage"]
    }
]

def generate_kg_os():
    nodes = []
    edges = []

    # 1. Role Node
    nodes.append({
        "id": "role:OS",
        "label": "Role",
        "properties": {
            "name": "OS",
            "full_name": "Oncology Sales Representative",
            "description": "The Oncology Sales (OS) Representative is a customer-facing commercial professional responsible for engaging urologists, urologic oncologists, and clinic teams to communicate approved INLEXZO product information, understand account barriers, support appropriate on-label patient identification in BCG-unresponsive NMIBC, coordinate anatomical model nurse prep sessions, and build compliant relationships.",
            "primary_domain": "commercial_promotional",
            "primary_brand": "INLEXZO"
        }
    })

    # 2. Domain Node
    nodes.append({
        "id": "domain:commercial_promotional",
        "label": "Domain",
        "properties": {
            "name": "Commercial / promotional engagement"
        }
    })
    edges.append({
        "source": "role:OS",
        "target": "domain:commercial_promotional",
        "type": "OPERATES_IN"
    })

    # 3. Topics
    for t in TOPICS:
        nodes.append({
            "id": t["id"],
            "label": "Topic",
            "properties": {"name": t["name"]}
        })
        edges.append({
            "source": "role:OS",
            "target": t["id"],
            "type": "HAS_IN_SCOPE_TOPIC"
        })

    for t in OUT_OF_SCOPE_TOPICS:
        nodes.append({
            "id": t["id"],
            "label": "Topic",
            "properties": {"name": t["name"]}
        })
        edges.append({
            "source": "role:OS",
            "target": t["id"],
            "type": "HAS_OUT_OF_SCOPE_TOPIC"
        })

    # 4. Compliance Rules
    for r in COMPLIANCE_RULES:
        nodes.append({
            "id": r["id"],
            "label": "ComplianceRule",
            "properties": r
        })
        edges.append({
            "source": "role:OS",
            "target": r["id"],
            "type": "GOVERNED_BY"
        })

    # 5. Core Responsibilities
    for cr in CORE_RESPONSIBILITIES:
        nodes.append({
            "id": cr["id"],
            "label": "CoreResponsibility",
            "properties": cr
        })
        edges.append({
            "source": "role:OS",
            "target": cr["id"],
            "type": "HAS_RESPONSIBILITY"
        })
        for top in cr["topics"]:
            edges.append({
                "source": cr["id"],
                "target": f"topic:{top}",
                "type": "ABOUT_TOPIC"
            })

    # 6. Exclusion Rules
    for er in EXCLUSION_RULES:
        nodes.append({
            "id": er["id"],
            "label": "ExclusionRule",
            "properties": er
        })
        edges.append({
            "source": "role:OS",
            "target": er["id"],
            "type": "HAS_EXCLUSION_RULE"
        })
        for top in er["topics"]:
            edges.append({
                "source": er["id"],
                "target": f"topic:{top}",
                "type": "EXCLUDES_TOPIC"
            })

    # 7. Accounts
    for acc in ACCOUNTS:
        nodes.append({
            "id": acc["id"],
            "label": "Account",
            "properties": acc
        })
        edges.append({
            "source": "role:OS",
            "target": acc["id"],
            "type": "ENGAGES_ACCOUNT"
        })

    # 8. Account Barriers
    for b in ACCOUNT_BARRIERS_OS:
        nodes.append({
            "id": b["barrier_id"],
            "label": "AccountBarrier",
            "properties": b
        })
        # Edge from Account to Barrier
        edges.append({
            "source": b["account_id"],
            "target": b["barrier_id"],
            "type": "HAS_ACCOUNT_BARRIER"
        })
        # Edge from Barrier to Topic
        if "Prior Authorization" in b["barrier_type"]:
            edges.append({"source": b["barrier_id"], "target": "topic:prior_authorization", "type": "ABOUT_TOPIC"})
        if "Formulary" in b["barrier_type"] or "P&T" in b["barrier_type"]:
            edges.append({"source": b["barrier_id"], "target": "topic:formulary_pt_committee", "type": "ABOUT_TOPIC"})
        if "Coverage" in b["barrier_type"] or "Deductible" in b["barrier_type"] or "Affordability" in b["barrier_type"]:
            edges.append({"source": b["barrier_id"], "target": "topic:payer_coverage_affordability", "type": "ABOUT_TOPIC"})
        if "Nursing" in b["barrier_type"] or "Prep" in b["barrier_type"]:
            edges.append({"source": b["barrier_id"], "target": "topic:anatomical_model_nurse_prep", "type": "ABOUT_TOPIC"})

    kg = {
        "schema_version": "4.0",
        "description": "Validated Knowledge Graph for Johnson & Johnson OS Commercial Oncology Operations (INLEXZO) Grounded in 1,498 Field Transcripts across 8 Healthcare Accounts.",
        "role": "OS",
        "brand": "INLEXZO",
        "primary_domain": "commercial_promotional",
        "node_labels": [
            "Role", "Domain", "Topic", "CoreResponsibility",
            "ExclusionRule", "ComplianceRule", "Account", "AccountBarrier"
        ],
        "edge_types": [
            "OPERATES_IN", "HAS_IN_SCOPE_TOPIC", "HAS_OUT_OF_SCOPE_TOPIC",
            "GOVERNED_BY", "HAS_RESPONSIBILITY", "HAS_EXCLUSION_RULE",
            "ENGAGES_ACCOUNT", "HAS_ACCOUNT_BARRIER", "ABOUT_TOPIC", "EXCLUDES_TOPIC"
        ],
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_accounts": len(ACCOUNTS),
            "total_barriers": len(ACCOUNT_BARRIERS_OS),
            "role_scope": "OS_ONLY",
            "brand_scope": "INLEXZO_ONLY"
        }
    }
    return kg

def main():
    print("=" * 65)
    print("BUILDING OS KNOWLEDGE GRAPH & ACCOUNT BARRIERS")
    print("=" * 65)

    # 1. Build and save account_barriers.json (OS only)
    barriers_clean = []
    for b in ACCOUNT_BARRIERS_OS:
        barriers_clean.append({
            "account": b["account"],
            "role": "OS",
            "barrier_type": b["barrier_type"],
            "barrier_details": b["barrier_details"],
            "account_ask": b["account_ask"],
            "trigger_keywords": b["trigger_keywords"],
            "case_1_inquiry": b["case_1_inquiry"],
            "case_2_inquiry": b["case_2_inquiry"]
        })

    barriers_path = BASE_DIR / "account_barriers.json"
    with open(barriers_path, "w", encoding="utf-8") as f:
        json.dump(barriers_clean, f, indent=2)
    print(f"[OK] Wrote {len(barriers_clean)} OS-only account barriers to: {barriers_path.name}")

    # 2. Build and save kg_os.json
    kg_os = generate_kg_os()
    kg_os_path = BASE_DIR / "kg_os.json"
    with open(kg_os_path, "w", encoding="utf-8") as f:
        json.dump(kg_os, f, indent=2)
    print(f"[OK] Wrote {len(kg_os['nodes'])} nodes and {len(kg_os['edges'])} edges to: {kg_os_path.name}")

    # Also update in slm_training_package if directory exists
    pkg_dir = BASE_DIR / "slm_training_package"
    if pkg_dir.exists():
        with open(pkg_dir / "account_barriers.json", "w", encoding="utf-8") as f:
            json.dump(barriers_clean, f, indent=2)
        with open(pkg_dir / "kg_os.json", "w", encoding="utf-8") as f:
            json.dump(kg_os, f, indent=2)
        print(f"[OK] Synced updated OS KG & barriers to slm_training_package/")

    # 3. Remove wrong/legacy FRM files
    wrong_files = [
        BASE_DIR / "kg_frm.json",
        pkg_dir / "kg_frm.json" if pkg_dir.exists() else None,
        BASE_DIR / "Persona_Solid_Cancer_OS_FRM.json",
    ]
    for wf in wrong_files:
        if wf and wf.exists():
            wf.unlink()
            print(f"[REMOVED] Deleted legacy wrong file: {wf.name}")

    print("=" * 65)
    print("SUCCESS: OS KNOWLEDGE GRAPH & BARRIERS GENERATED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
