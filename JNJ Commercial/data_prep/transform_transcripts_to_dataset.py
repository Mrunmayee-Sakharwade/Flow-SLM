"""
Data Transformation Engine: Excel Transcripts to Structured Turn Dataset
=======================================================================
Transforms 2,000 Excel field transcripts (1k OS + 1k FRM) into a structured
retrieval and state dataset matching the canonical format:

    current_question -> candidate_answer -> state -> kg_topics -> next_question

Outputs:
  - data_prep/training_turns_dataset.json (Comprehensive full dataset)
  - data_prep/dataset_summary.json (Dataset statistics & state distributions)
"""

import os
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 1. KG Ontology & Topic Lexicons
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    "prior authorization": [
        "prior auth", "prior authorization", "pa ", "pa's", "pa coordinator", 
        "appeal", "appeals", "denial", "denials", "approval", "pre-auth", "medical necessity"
    ],
    "payer coverage": [
        "payer", "payers", "commercial plan", "medicare", "medicaid", "coverage", 
        "policy", "formulary", "tier", "restriction", "site-of-care", "step therapy"
    ],
    "affordability patient support": [
        "copay", "co-pay", "pap", "patient assistance", "foundation", "financial assistance", 
        "affordability", "out-of-pocket", "cost barrier", "quick start", "bridge program"
    ],
    "reimbursement": [
        "reimbursement", "billing", "claim", "claims", "j-code", "coding", "buy-and-bill", 
        "hub", "enrollment", "specialty pharmacy", "sp ", "remittance"
    ],
    "patient access": [
        "access", "patient access", "access barrier", "access check-in", "intake", 
        "distribution", "fulfillment"
    ],
    "biomarker testing": [
        "biomarker", "biomarkers", "egfr", "exon 20", "ngs", "testing turnaround", 
        "liquid biopsy", "tissue biopsy", "molecular testing"
    ],
    "efficacy safety product info": [
        "efficacy", "clinical trial", "pfs", "overall survival", "response rate", 
        "indication", "clinical evidence", "label", "monotherapy", "combination", "lazcluze"
    ],
    "dosing administration": [
        "dosing", "infusion", "administration", "weight-based", "subcutaneous", 
        "dose modification", "interruption", "reconstitution"
    ],
    "treatment sequencing": [
        "first-line", "second-line", "1l", "2l", "sequencing", "post-progression", 
        "prior therapy", "treatment algorithm", "prescribing preference"
    ],
    "toxicity management": [
        "toxicity", "adverse event", "infusion reaction", "irr", "rash", "paronychia", 
        "safety concern", "mitigation protocol", "prophylaxis"
    ],
    "clinical operational workflow": [
        "order set", "ehr", "electronic ordering", "template", "room setup", 
        "anatomical demo", "insertion", "device", "procedure room", "workflow"
    ],
    "stakeholder management": [
        "oncologist", "urologist", "nurse", "lead rn", "app", "physician assistant", 
        "nurse practitioner", "practice manager", "reimbursement specialist", "billing team"
    ],
    "cross functional collaboration": [
        "psm", "msl", "frm", "os", "kam", "medical affairs", "account support", 
        "cross-functional", "handoff", "loop in"
    ],
    "territory engagement planning": [
        "follow-up", "next steps", "email", "meeting", "visit", "schedule", 
        "action plan", "target timing"
    ]
}

# ---------------------------------------------------------------------------
# 2. State Classification Logic
# ---------------------------------------------------------------------------
def classify_state(turn_idx, total_turns, q_curr, a_curr, q_next, role):
    """
    Classifies the dialogue turn into a canonical FSM State based on
    current question, candidate answer, position in conversation, and role.
    """
    q_curr_lower = q_curr.lower() if q_curr else ""
    q_next_lower = q_next.lower() if q_next else ""
    a_curr_lower = a_curr.lower() if a_curr else ""
    
    # State 0: Greeting / Initiation
    if turn_idx == 0 or "do you have a minute" in q_curr_lower or "ready to capture" in q_curr_lower:
        return "STATE_0_GREETING_INITIATION"
    
    # State 7: Wrap-up / Final confirmation
    if q_next is None or "anything else before" in q_curr_lower or "close out this" in q_curr_lower or "ready to wrap" in q_curr_lower or "ready to end" in q_curr_lower or "should we wrap" in q_curr_lower:
        return "STATE_7_WRAP_UP_CONFIRMATION"
    
    # State 1: Account & Stakeholder Identification
    if "who did you meet" in q_curr_lower or "who was the call with" in q_curr_lower or "where was the account" in q_curr_lower:
        return "STATE_1_ACCOUNT_STAKEHOLDER"
    
    # State 2: Primary Purpose / Initial Call Focus
    if "what prompted" in q_curr_lower or "main purpose" in q_curr_lower or "main access topic" in q_curr_lower or "what inlexzo" in q_curr_lower or "what rybrevant" in q_curr_lower:
        return "STATE_2_PRIMARY_PURPOSE"
    
    # State 6: Next Actions & Ownership
    if "who owns the next" in q_curr_lower or "who owns the follow" in q_curr_lower or "next access action" in q_curr_lower or "follow-up plan" in q_curr_lower or "target timing" in q_curr_lower or "when are they hoping" in q_curr_lower:
        return "STATE_6_NEXT_ACTIONS"
    
    # State 5: Cross-Functional Collaboration / Handoffs
    if "psm" in q_curr_lower or "msl" in q_curr_lower or "account support" in q_curr_lower or "loop in" in q_curr_lower or "collaborate" in q_curr_lower:
        return "STATE_5_CROSS_FUNCTIONAL_HANDOFF"
    
    # State 4: Barriers / Logistics / Readiness Friction
    if "barrier" in q_curr_lower or "logistics" in q_curr_lower or "readiness" in q_curr_lower or "workflow step" in q_curr_lower or "messy" in q_curr_lower:
        return "STATE_4_BARRIERS_LOGISTICS"
    
    # State 3: Deep Dive (Role Specific)
    if role == "FRM":
        if "payer" in q_curr_lower or "prior auth" in q_curr_lower or "denial" in q_curr_lower or "appeal" in q_curr_lower or "claim" in q_curr_lower:
            return "STATE_3A_PRIOR_AUTH_PAYER"
        elif "affordability" in q_curr_lower or "copay" in q_curr_lower or "pap" in q_curr_lower or "assistance" in q_curr_lower:
            return "STATE_3B_AFFORDABILITY_COPAY_PAP"
        elif "hub" in q_curr_lower or "specialty pharmacy" in q_curr_lower or "enrollment" in q_curr_lower:
            return "STATE_3C_HUB_SPECIALTY_PHARMACY"
        else:
            return "STATE_3_REIMBURSEMENT_DETAILS"
    else: # OS Role
        if "workflow" in q_curr_lower or "demo" in q_curr_lower or "room setup" in q_curr_lower:
            return "STATE_3E_WORKFLOW_DEMO_REFRESHER"
        elif "efficacy" in q_curr_lower or "considering" in q_curr_lower or "positioning" in q_curr_lower or "treatment" in q_curr_lower:
            return "STATE_3F_EFFICACY_TREATMENT_POSITIONING"
        elif "eligible patient" in q_curr_lower or "patient selection" in q_curr_lower:
            return "STATE_3G_PATIENT_IDENTIFICATION"
        else:
            return "STATE_3_CLINICAL_PROMOTIONAL_DETAILS"


# ---------------------------------------------------------------------------
# 3. Topic Tagging & Scope Validation
# ---------------------------------------------------------------------------
def extract_topics(text):
    text_lower = text.lower()
    matched = []
    for topic, kws in TOPIC_KEYWORDS.items():
        for kw in kws:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                matched.append(topic)
                break
    return matched

def check_scope(role, topics, out_of_scope_dict):
    out_scope = out_of_scope_dict.get(role, set())
    violations = [t for t in topics if t in out_scope]
    return {
        "is_in_scope": len(violations) == 0,
        "violations": violations
    }

# ---------------------------------------------------------------------------
# 4. XML-Based Excel Reader (High Performance, 0 external deps)
# ---------------------------------------------------------------------------
def read_xlsx(filename):
    with zipfile.ZipFile(filename, 'r') as z:
        sst = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
            sst = [''.join(node.itertext()) for node in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si')]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = sheet.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
        
        all_records = []
        for r_idx, row in enumerate(rows[1:]): # skip header
            row_vals = []
            for c in row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                t = c.get('t')
                v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                val = v.text if v is not None else ''
                if t == 's' and val.isdigit():
                    val = sst[int(val)]
                row_vals.append(val)
            if len(row_vals) >= 5:
                all_records.append({
                    's_no': row_vals[0],
                    'brand': row_vals[1],
                    'role': row_vals[2],
                    'name': row_vals[3],
                    'text': row_vals[4]
                })
        return all_records

# ---------------------------------------------------------------------------
# 5. Main Dataset Transformation Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STARTING TRANSCRIPT-TO-DATASET TRANSFORMATION PIPELINE")
    print("=" * 70)
    
    # Load KG for Scope Definitions
    with open('Persona_Solid_Cancer_OS_FRM.json', 'r') as f:
        kg = json.load(f)
    
    out_of_scope_dict = defaultdict(set)
    in_scope_dict = defaultdict(set)
    
    for edge in kg['edges']:
        if edge['type'] == 'HAS_OUT_OF_SCOPE_TOPIC':
            role = edge['source'].replace('role:', '')
            topic = edge['target'].replace('topic:', '').replace('_', ' ')
            out_of_scope_dict[role].add(topic)
        elif edge['type'] == 'HAS_IN_SCOPE_TOPIC':
            role = edge['source'].replace('role:', '')
            topic = edge['target'].replace('topic:', '').replace('_', ' ')
            in_scope_dict[role].add(topic)
    
    print(f"Loaded KG Scopes: OS out-of-scope={len(out_of_scope_dict['OS'])}, FRM out-of-scope={len(out_of_scope_dict['FRM'])}")
    
    frm_records = read_xlsx('generated_frm_training_transcripts_1k.xlsx')
    os_records = read_xlsx('generated_os_training_transcripts_1k.xlsx')
    
    print(f"Loaded {len(frm_records)} FRM dialogues and {len(os_records)} OS dialogues.")
    
    all_turns_dataset = []
    state_distribution = Counter()
    topic_distribution = Counter()
    role_distribution = Counter()
    
    for dataset, default_role in [(frm_records, "FRM"), (os_records, "OS")]:
        for doc_idx, doc in enumerate(dataset):
            dialogue_id = f"{default_role}_{int(doc['s_no']):04d}"
            brand = doc['brand']
            role = doc['role'] if doc['role'] else default_role
            rep_name = doc['name']
            
            # Parse raw transcript into turns
            raw_lines = [l.strip() for l in doc['text'].split('\n') if l.strip()]
            turns = []
            curr_speaker = None
            curr_text = []
            
            for line in raw_lines:
                if line.startswith('AI:') or line.startswith('User:'):
                    if curr_speaker:
                        turns.append((curr_speaker, ' '.join(curr_text)))
                    curr_speaker = 'AI' if line.startswith('AI:') else 'User'
                    curr_text = [line.split(':', 1)[1].strip()]
                else:
                    curr_text.append(line)
            if curr_speaker:
                turns.append((curr_speaker, ' '.join(curr_text)))
            
            # Generate (Q_t, A_t, State, Topics, Q_{t+1}) records
            qa_pairs = []
            for i in range(0, len(turns)-1, 2):
                q_t = turns[i][1] if turns[i][0] == 'AI' else ""
                a_t = turns[i+1][1] if i+1 < len(turns) and turns[i+1][0] == 'User' else ""
                q_next = turns[i+2][1] if i+2 < len(turns) and turns[i+2][0] == 'AI' else None
                if q_t and a_t:
                    qa_pairs.append((q_t, a_t, q_next))
            
            total_qa = len(qa_pairs)
            for turn_idx, (q_t, a_t, q_next) in enumerate(qa_pairs):
                # Detect Topics in User Answer
                topics = extract_topics(a_t)
                scope_info = check_scope(role, topics, out_of_scope_dict)
                
                # Classify State
                state = classify_state(turn_idx, total_qa, q_t, a_t, q_next, role)
                
                # Record turn
                turn_record = {
                    "turn_id": f"{dialogue_id}_T{turn_idx:02d}",
                    "dialogue_id": dialogue_id,
                    "brand": brand,
                    "role": role,
                    "rep_name": rep_name,
                    "turn_index": turn_idx,
                    "total_turns": total_qa,
                    "current_question": q_t,
                    "candidate_answer": a_t,
                    "state": state,
                    "kg_topics": topics,
                    "is_in_scope": scope_info["is_in_scope"],
                    "scope_violations": scope_info["violations"],
                    "next_question": q_next,
                    "is_terminal": q_next is None
                }
                
                all_turns_dataset.append(turn_record)
                state_distribution[state] += 1
                role_distribution[role] += 1
                for top in topics:
                    topic_distribution[top] += 1

    # Save to disk
    output_json_path = os.path.join('data_prep', 'training_turns_dataset.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_turns_dataset, f, indent=2)
    
    summary_path = os.path.join('data_prep', 'dataset_summary.json')
    summary_data = {
        "total_dialogues": len(frm_records) + len(os_records),
        "total_turns_extracted": len(all_turns_dataset),
        "turns_by_role": dict(role_distribution),
        "state_distribution": dict(state_distribution.most_common()),
        "top_topics_detected": dict(topic_distribution.most_common())
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRANSFORMATION COMPLETE!")
    print(f"Total Structured Turns Generated: {len(all_turns_dataset):,}")
    print(f"  - FRM Turns: {role_distribution['FRM']:,}")
    print(f"  - OS Turns:  {role_distribution['OS']:,}")
    print(f"Saved dataset to: {output_json_path}")
    print(f"Saved summary to: {summary_path}")
    print("\nState Distribution:")
    for st, cnt in state_distribution.most_common():
        print(f"  {st:38s}: {cnt:5d} turns")
    print("=" * 70)

if __name__ == "__main__":
    main()
