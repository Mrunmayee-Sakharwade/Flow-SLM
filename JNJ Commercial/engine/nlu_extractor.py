"""
NLU & Entity/Topic Extractor (Zero-LLM Stack)
==============================================
Extracts domain entities (HCPs, clinics, payers, stakeholders, brands) and
detects KG topic matches using dictionary gazetteers and rule-based regex patterns.
"""

import os
import re
from typing import Dict, List, Any

# Medical & Oncology Gazetteers
BRANDS = ["RYBREVANT", "INLEXZO", "LAZCLUZE"]
STAKEHOLDER_ROLES = [
    "oncologist", "urologist", "lead rn", "registered nurse", "nurse", 
    "app", "physician assistant", "nurse practitioner", "practice manager", 
    "prior auth coordinator", "reimbursement coordinator", "billing coordinator",
    "financial counselor", "specialty pharmacy"
]
PAYER_TYPES = [
    "commercial payer", "medicare", "medicaid", "medicare advantage", 
    "blue cross", "aetna", "cigna", "united healthcare", "humana"
]

CANONICAL_ACCOUNTS = {
    "apollo": "Apollo Hospitals",
    "fortis": "Fortis Healthcare",
    "manipal": "Manipal Hospitals",
    "max": "Max Healthcare",
    "narayana": "Narayana Health"
}

TOPIC_PATTERNS = {
    "account identification": [
        r"\b(hospital|clinic|cancer\s+center|medical\s+center|health\s+system|health\s+network|institute|account|office|site|practice|infusion\s+center|department|apollo|fortis|manipal|max\s+healthcare|narayana)\b"
    ],
    "stakeholder management": [
        r"\b(dr\.?|doctor|md\b|do\b|oncologist|urologist|nurse|rn|np\b|pa\b|physician|coordinator|manager|staff|team|counselor|specialist|practitioner|administrator)\b"
    ],
    "prior authorization": [
        r"\b(prior\s+auth|prior\s+authorization|pa\b|pa\'s|appeal|denial|denials|appeals|pre-auth)\b"
    ],
    "payer coverage": [
        r"\b(payer|payers|coverage|formulary|tier|policy|site-of-care|step\s+therapy)\b"
    ],
    "affordability patient support": [
        r"\b(copay|co-pay|pap|patient\s+assistance|foundation|financial\s+assistance|affordability|out-of-pocket)\b"
    ],
    "reimbursement": [
        r"\b(reimbursement|billing|claim|claims|j-code|coding|buy-and-bill|hub|enrollment|specialty\s+pharmacy)\b"
    ],
    "patient access": [
        r"\b(patient\s+access|access\s+barrier|access\s+check-in|access\s+support)\b"
    ],
    "biomarker testing": [
        r"\b(biomarker|biomarkers|egfr|exon\s*20|ngs|molecular\s+testing|turnaround\s+time)\b"
    ],
    "efficacy safety product info": [
        r"\b(nmibc|bladder|baldder|bcg|bcg-unresponsive|bcg\s+failure|cancer|carcinoma|urothelial|tumor|solid\s+tumor|efficacy|clinical\s+trial|pfs|overall\s+survival|response\s+rate|indication|clinical\s+evidence|label|monotherapy|combination|eligible|eligibility|patient\s+selection|patient\s+identification|disease\s+state)\b"
    ],
    "dosing administration": [
        r"\b(dosing|infusion|administration|subcutaneous|dose\s+modification|interruption|reconstitution)\b"
    ],
    "treatment sequencing": [
        r"\b(first-line|second-line|1l|2l|sequencing|post-progression|post\s+bcg|bcg\s+failure|treatment\s+pathway|treatment\s+algorithm|prescribing\s+preference|eligible\s+patients?|eligible\s+pateintents?|candidate|candidates)\b"
    ],
    "toxicity management": [
        r"\b(toxicity|adverse\s+event|infusion\s+reaction|irr|rash|paronychia|safety\s+concern|mitigation\s+protocol)\b"
    ],
    "clinical operational workflow": [
        r"\b(order\s+set|ehr|template|room\s+setup|anatomical\s+demo|insertion|device|procedure\s+room|workflow)\b"
    ],
    "cross functional collaboration": [
        r"\b(psm|msl|frm|os|kam|medical\s+affairs|account\s+support|cross-functional|handoff|loop\s+in)\b"
    ],
    "territory engagement planning": [
        r"\b(follow-up|next\s+steps|email|meeting|visit|schedule|action\s+plan|target\s+timing)\b"
    ]
}

class NLUExtractor:
    def __init__(self):
        self.brands = BRANDS
        self.roles = STAKEHOLDER_ROLES
        self.payers = PAYER_TYPES
        
    def extract_topics(self, text: str) -> List[str]:
        """Extracts KG topic keys present in the user text."""
        detected = []
        text_lower = text.lower()
        for topic, patterns in TOPIC_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    detected.append(topic)
                    break
        return detected

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extracts named entities from the candidate answer using gazetteers and patterns."""
        entities = {
            "brands": [],
            "hcps": [],
            "accounts": [],
            "stakeholder_roles": [],
            "payers": [],
            "dates_or_timings": []
        }
        
        # Brands
        for b in self.brands:
            if re.search(r'\b' + re.escape(b) + r'\b', text, re.IGNORECASE):
                if b not in entities["brands"]:
                    entities["brands"].append(b)
        if any(w in text.lower() for w in ["nmibc", "bladder", "baldder", "bcg", "bcg failure", "post bcg", "tar-200", "instillation", "insertion"]):
            if "INLEXZO" not in entities["brands"]:
                entities["brands"].append("INLEXZO")
        if any(w in text.lower() for w in ["mariposa", "1l egfr", "first-line egfr"]):
            if "RYBREVANT + LAZCLUZE" not in entities["brands"]:
                entities["brands"].append("RYBREVANT + LAZCLUZE")
                
        # HCP Names (e.g., Dr. Robert Patel, Dr Smith, Doctor Avery, met with Sarah)
        hcp_matches = re.findall(r'\b(?:Dr\.?|Doctor)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b', text, re.IGNORECASE)
        if not hcp_matches:
            context_matches = re.findall(r'\b(?:met with|spoke with|visited|called|saw)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b', text, re.IGNORECASE)
            for m in context_matches:
                m_low = m.lower()
                if not any(k in m_low for k in CANONICAL_ACCOUNTS) and not any(w in m_low for w in ["hospital", "clinic", "cancer", "center", "centre", "dr", "doctor", "team", "staff", "the", "a", "an", "health", "oncology", "urology"]):
                    hcp_matches.append(m)

        # Standardize prefix as "Dr. <Name>" for HCP entity
        for h in hcp_matches:
            clean_name = re.sub(r'\s+(?:at|in|on|with|from|to|regarding|about|for|and|during|re|hospital|clinic|raised|shared|mentioned|asked|noted|discussed|said|stated|agreed|indicated|wants|explained|expressed|reported|has|had|is|was)$', '', h.strip(), flags=re.IGNORECASE)
            clean_name = re.sub(r'^(?:at|in|on|with|from|to|regarding|about|for)\s+', '', clean_name.strip(), flags=re.IGNORECASE)
            clean_name = " ".join(part.capitalize() for part in clean_name.split())
            formatted_hcp = clean_name if clean_name.startswith("Dr.") else f"Dr. {clean_name}"
            if formatted_hcp not in entities["hcps"] and formatted_hcp != "Dr.":
                entities["hcps"].append(formatted_hcp)

        # Canonical Major Healthcare Accounts Matching
        for key, canonical_name in CANONICAL_ACCOUNTS.items():
            if re.search(r'\b' + re.escape(key) + r'(\s+(?:hospitals?|healthcare|health|clinic|medical))?\b', text, re.IGNORECASE):
                if canonical_name not in entities["accounts"]:
                    entities["accounts"].append(canonical_name)

        # Account / Hospital / Clinic extraction (e.g. City Cancer Center, Memorial Hospital, Mayo Clinic)
        account_matches = re.findall(r'\b([A-Za-z0-9\']+(?:\s+[A-Za-z0-9\']+)?\s+(?:Hospital|Clinic|Cancer Center|Medical Center|Health|Institute|Center|Centre|System|Network))\b', text, re.IGNORECASE)
        if not account_matches:
            loc_matches = re.findall(r'\b(?:at|in)\s+([A-Za-z0-9\']+(?:\s+Hospital|\s+Clinic|\s+Center|\s+Centre)?)\b', text, re.IGNORECASE)
            for lm in loc_matches:
                if lm.lower() not in ("the", "a", "an", "friday", "tuesday", "thursday", "monday", "wednesday", "saturday", "sunday", "december", "november", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "dr", "doctor"):
                    account_matches.append(lm)

        for acc in account_matches:
            clean_acc = re.sub(r'^(?:at|in|on|with|from|to|the|a|an|our)\s+', '', acc.strip(), flags=re.IGNORECASE)
            clean_acc = " ".join(part.capitalize() for part in clean_acc.split())
            if clean_acc.lower() in ["clinic", "hospital", "center", "centre", "cancer center", "medical center", "team", "staff"]:
                continue
            # Normalize to canonical if recognized
            matched_canonical = None
            for key, canonical_name in CANONICAL_ACCOUNTS.items():
                if key in clean_acc.lower():
                    matched_canonical = canonical_name
                    break
            final_acc = matched_canonical or clean_acc
            if final_acc not in entities["accounts"] and final_acc:
                entities["accounts"].append(final_acc)
        
        # Stakeholder Roles
        text_lower = text.lower()
        for r in self.roles:
            if re.search(r'\b' + re.escape(r) + r'\b', text_lower):
                entities["stakeholder_roles"].append(r)
                
        # Payers
        for p in self.payers:
            if re.search(r'\b' + re.escape(p) + r'\b', text_lower):
                entities["payers"].append(p)
                
        # Timing / Dates
        timing_matches = re.findall(r'\b(by\s+[A-Za-z]+|early\s+[A-Za-z]+|week\s+of\s+[A-Za-z]+\s+\d+|before\s+[A-Za-z]+|Tuesday|Thursday|Friday|Monday|Wednesday)\b', text, re.IGNORECASE)
        entities["dates_or_timings"].extend(timing_matches)
        
        return entities

    def check_out_of_domain(self, text: str) -> Dict[str, Any]:
        """
        Detects if the rep input is an out-of-domain / irrelevant query or unrelated nonsense.
        Active by default to guard call notes against non-commercial / off-topic content.
        """
        text_clean = text.strip().lower()
        if not text_clean:
            return {"is_out_of_domain": True, "suggested_message": "I didn't receive any input. Could you please share your call details?"}

        # 1. Affirmations & Positive Conversational Replies (e.g., 'yes sure', 'yes please', 'sure thing', 'we can')
        affirmation_patterns = [
            r'^(yes|yep|yeah|sure|ok|okay|of\s+course|certainly|absolutely|definitely|ready|go\s+ahead|lets\s+do\s+it|let\'s\s+do\s+it|lets\s+start|let\'s\s+start|lets\s+go|let\'s\s+go|i\s+do|i\s+have\s+time|yeah\s+sure|yes\s+sure|yes\s+please|sure\s+thing|sure\s+go\s+ahead|yes\s+let\'s\s+do\s+it|yes\s+lets\s+do\s+it|yes\s+of\s+course|yes\s+i\s+do|why\s+not|all\s+set|sounds\s+good|proceed|let\'s\s+begin|fine|we\s+can|yes\s+we\s+can|yep\s+we\s+can)[.! ]*$',
            r'\b(yes|yeah|yep|sure\s+thing|go\s+ahead|of\s+course|certainly|let\'s\s+do\s+it|we\s+can|yes\s+we\s+can)\b'
        ]
        for ap in affirmation_patterns:
            if re.search(ap, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 2. Negations & Session Control Commands (e.g., 'no', 'not now', 'busy', 'nothing else', 'done')
        negation_patterns = [
            r'^(no|nope|nah|none|nothing|nothing\s+else|no\s+more(\s+notes)?|no\s+other\s+topics|nothing\s+to\s+(add|report)|done|finished|all\s+set|stop|exit|cancel|not\s+now|busy|later|don\'t\s+have\s+time|no\s+time|i\s+want\s+to\s+end|n/a|no\s+action|no\s+updates|bye|goodbye)[.! ]*$',
            r'\b(no\s+more|not\s+now|busy\s+right\s+now|later|don\'t\s+have\s+time|no\s+time|nothing\s+else|nothing\s+to\s+add|there\s+arent|there\s+aren\'t|no\s+barriers?|no\s+issues?|none\s+identified)\b',
            r'\b(no\s+there\s+aren\'t(\s+any)?|there\s+aren\'t\s+any|none\s+at\s+all|not\s+really|not\s+at\s+all)\b',
            r'\b(did\s+not\s+mention|didn\'t\s+mention|did\s+not\s+say|didn\'t\s+say|not\s+mention|nothing\s+at\s+the\s+moment|not\s+at\s+the\s+moment|not\s+at\s+this\s+time|not\s+right\s+now)\b',
            r'\b(this\s+is\s+all(\s+we\s+have|\s+i\s+have)?|that\'s\s+all(\s+we\s+have|\s+i\s+have)?|that\s+is\s+all|all\s+we\s+have|all\s+i\s+have|nothing\s+else|that\s+would\s+be\s+all)\b',
            r'\b(he\s+did\s+not|she\s+did\s+not|they\s+did\s+not|doctor\s+did\s+not)\b'
        ]
        for np in negation_patterns:
            if re.search(np, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 3. Healthcare, Oncology, Access & Commercial Terminology
        healthcare_whitelist = [
            r"\b(dr\.?|doctor|md\b|do\b|physician|patient|patients|oncology|urology|hospital|clinic|center|centre|institute|account|office|site|practice|health\s+system|infusion\s+center)\b",
            r"\b(inlexzo|rybrevant|lazcluze|brand|drug|therapy|regimen|nsclc|nmibc|bcg|bcg-unresponsive|bladder|baldder|urothelial|tarceva|tagrisso|chemo|immunotherapy|medication|product)\b",
            r"\b(prior\s+auth|pa\b|copay|co-pay|pap|hub|access|reimbursement|payer|medicare|medicaid|blue\s+cross|aetna|cigna|united|coverage|appeal|appeals|denial|denials|denied)\b",
            r"\b(dosing|infusion|administration|subcutaneous|efficacy|trial|pfs|safety|rash|adverse|toxicity|mitigation|workflow|ehr|order\s+set|protocol|support|program|enrollment|form|formulary|barrier|delay|turnaround|approved|approval|pending|submitted|specialty\s+pharmacy|distributor|billing|claim|claims|j-code|code|coding)\b",
            r"\b(nurse|rn|np\b|pa\b|coordinator|manager|staff|counselor|specialist|rep|kam|msl|frm|os|mir|loop\s+in|sample|samples|brochure|materials|slide|deck)\b",
            r"\b(post\s+bcg|bcg\s+failure|unresponsive|cystectomy|medical\s+history|suitable|suitability|treatment\s+pathway|treatment\s+approach|eligible|eligibility)\b"
        ]
        for hw in healthcare_whitelist:
            if re.search(hw, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 4. Meeting Discourse & Commercial Collaboration Verbs
        meeting_discourse = [
            r"\b(met\s+with|visited|spoke\s+with|called|follow[\s\-_]*up|next\s+step|action\s+plan|discussed|talked|reviewed|shared|agreed|presented|explained|asked|question|inquired|clarified|educated|scheduled|meeting|call|visit|check-in|outreach|touchpoint)\b",
            r"\b(everything\s+(fine|good|clear|smooth|set)|no\s+(issue|issues|problem|questions|barriers)|looking\s+into|working\s+on|in\s+progress|not\s+yet|already|waiting)\b",
            r"\b(next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|tomorrow|by\s+(friday|tuesday|monday|wednesday|thursday)|early\s+[a-z]+|couple\s+of\s+weeks|weeks|days|month|months)\b",
            r"\b(aiming\s+to|planning\s+to|decide|deciding|reviewing|checking|updating|update|check\s+the\s+complete)\b"
        ]
        for md in meeting_discourse:
            if re.search(md, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 5. Check if known entities were extracted
        entities = self.extract_entities(text)
        if any(len(v) > 0 for v in entities.values()):
            return {"is_out_of_domain": False}

        # 6. Check if known topics were detected
        topics = self.extract_topics(text)
        if topics:
            return {"is_out_of_domain": False}

        # If it matched NONE of the above domain criteria (e.g. 'Ice cream', 'weather', 'joke', etc.)
        return {
            "is_out_of_domain": True,
            "reason": "Unrecognized or non-commercial input",
            "suggested_message": "That appears out of context for this call note."
        }

    def analyze_utterance(self, text: str) -> Dict[str, Any]:
        """Performs complete NLU pass on raw utterance."""
        text_clean = text.strip().lower()
        is_neg = False
        if re.match(r'^(no|nope|nah|none|nothing|nothing\s+else|no\s+more(\s+notes)?|no\s+other\s+topics|nothing\s+to\s+(add|report)|done|not\s+now|busy|later|no\s+action)[.! ]*$', text_clean, re.IGNORECASE):
            is_neg = True
            
        return {
            "text": text,
            "topics": self.extract_topics(text),
            "entities": self.extract_entities(text),
            "is_empty_or_negative": is_neg
        }
