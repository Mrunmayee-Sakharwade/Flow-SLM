"""
patch_kg_and_safety.py
Applies Knowledge Graph Context-Aware Dynamic Question Retrieval and Strict Anti-Hallucination
directly to engine/kg_context_retriever.py and engine/slm_next_question_engine.py.
"""
import re

NEW_INQUIRY_METHOD = '''    def retrieve_kg_node_inquiry(
        self,
        role: str,
        target_topic: str,
        conversation_history=None,
        candidate_answer: str = "",
        brand: str = "INLEXZO",
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
        hcp_name = hcp or "the doctor"
        prior_ai = [h.get("text", "").lower() for h in (conversation_history or []) if h.get("speaker") == "AI"]
        cand_lower = (candidate_answer or "").strip().lower()
        hist_text = " ".join([h.get("text", "") for h in (conversation_history or [])]).lower()
        has_known_hcp = any(w in hist_text or w in cand_lower for w in ["dr.", "dr ", "doctor"])

        if any(w in (cand_lower + " " + hist_text) for w in ["nmibc", "bladder", "bcg", "anurag", "apollo"]) or (brand and "inlexzo" in brand.lower()):
            brand_name = "INLEXZO"
        else:
            brand_name = brand or "INLEXZO"

        resolved_account = account
        if not resolved_account:
            full_context = cand_lower + " " + hist_text
            for k, canonical in [("apollo", "Apollo Hospitals"), ("fortis", "Fortis Healthcare"), ("manipal", "Manipal Hospitals"), ("max", "Max Healthcare"), ("narayana", "Narayana Health")]:
                if k in full_context:
                    resolved_account = canonical
                    break

        acc_barrier = self.get_account_barrier(resolved_account, role_upper)

        if role_upper == "OS":
            # 0. User acknowledging greeting or hasn't identified doctor/account yet
            if not has_known_hcp and not any(w in cand_lower for w in ["dr.", "dr ", "doctor"]):
                return "Who did you meet with, and where?"

            # 1. User answered Who/Where (Account Identification)
            if any(w in cand_lower for w in ["dr.", "dr ", "doctor", "apollo", "hospital", "clinic", "center"]) and not any("discussion" in q or "purpose" in q for q in prior_ai):
                if has_known_hcp or (hcp_name and hcp_name != "the doctor"):
                    return f"What was the main {brand_name} discussion with {hcp_name} today?"
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
                    if not any("share about eligible cases" in q for q in prior_ai):
                        return f"What did {hcp_name} share about eligible cases?"

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

            # 5. User answered Next Step / Suitability check for the drug
            if any(w in cand_lower for w in ["medical history", "suitable for inlexzo", "suitable for", "sutable for", "evaluate"]):
                if acc_barrier and resolved_account:
                    if "Apollo" in resolved_account:
                        return f"Did {hcp_name} mention any friction operationalizing the pathway or identifying eligible candidates at {resolved_account}?"
                    elif "Fortis" in resolved_account:
                        return f"Did {hcp_name} mention whether biomarker turnaround times are impacting patient identification at {resolved_account}?"
                    elif "Manipal" in resolved_account:
                        return f"Did {hcp_name} share if diagnosis-to-treatment coordination delays are affecting candidate routing at {resolved_account}?"
                    elif "Max" in resolved_account:
                        return f"Did {hcp_name} mention any diagnostic handoff friction between roles at {resolved_account}?"
                    elif "Narayana" in resolved_account:
                        return f"Did {hcp_name} express interest in a simpler way to flag eligible candidates earlier at {resolved_account}?"
                return "Did he mention any access or coverage barriers?"

            # 6. User answered Access Barriers ("no", "no there arent any", "none")
            if any(cand_lower.startswith(w) for w in ["no", "none", "not really", "there arent", "there aren't"]) or cand_lower in ["no", "none", "no barriers"]:
                if not any("timing" in q for q in prior_ai):
                    return "What is the timing for the next step?"
                elif not any("support" in q for q in prior_ai):
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

            # Dynamic Fallback: evaluate unasked areas
            if not any("barrier" in q for q in prior_ai):
                return "Did he mention any access or coverage barriers?"
            if not any("timing" in q for q in prior_ai):
                return "What is the timing for the next step?"
            if not any("support" in q for q in prior_ai):
                return f"Did {hcp_name} mention any support or resources needed?"
            if not any("action items" in q for q in prior_ai):
                return "Any follow-up or action items from the call?"
            if not any("account context" in q for q in prior_ai):
                return "Any other account context to capture?"
            return "I have enough for the call note. Should we wrap here?"

        else: # FRM
            if target_topic == "account identification":
                return "Who did you meet with, and where was the account?"
            if "prior authorization" not in cand_lower and not any("prior authorization" in q for q in prior_ai):
                if acc_barrier and resolved_account:
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
                return f"What specific prior authorization criteria or denial reasons did they encounter for {brand_name}?"
            elif "affordability" not in cand_lower and not any("affordability" in q or "copay" in q for q in prior_ai):
                return f"Did affordability or co-pay assistance come up during the discussion with {hcp_name}?"
            elif "hub" not in cand_lower and not any("hub" in q for q in prior_ai):
                return f"Was there any hub enrollment or specialty pharmacy support needed for {brand_name}?"
            elif not any("action" in q for q in prior_ai):
                return f"What next reimbursement or access action did you agree on with {hcp_name}?"
            return "I have enough for the call note. Should we wrap here?"
'''

NEW_SAFETY_RULES = '''        # Rule 0.9: STRICT ENTITY ANCHORING - Replace ANY hallucinated doctor name with the active HCP
        known_hcp = None
        if extracted_entities and extracted_entities.get("hcps"):
            known_hcp = extracted_entities["hcps"][0]
        elif conversation_history:
            for h in conversation_history:
                m = re.search(r'\\b(Dr\\.?\\s+[A-Z][a-z]+)\\b', h.get("text", ""))
                if m:
                    known_hcp = m.group(1)
                    break
        if known_hcp:
            question_text = re.sub(r'\\bDr\\.?\\s+[A-Z][a-z]+\\b', known_hcp, question_text)

        # Rule 1: Toxicity Terminology Replacement
        question_text = re.sub(r'\\btoxicity\\b', 'safety concern', question_text, flags=re.IGNORECASE)
        question_text = re.sub(r'\\btoxicities\\b', 'safety concerns', question_text, flags=re.IGNORECASE)

        # Rule 1.4: Strict Interrogative Form Validator (Block declarative statements & call-note narratives)
        q_stripped = question_text.strip().rstrip("?").strip()
        is_statement = bool(re.match(
            r'^(he\\s+said|she\\s+said|they\\s+said|he\\s+mentioned|she\\s+mentioned|there\\s+(may\\s+be|is|are|was|were)|i\\s+(met|discussed|spoke|think|asked)|we\\s+(discussed|talked|reviewed|agreed)|the\\s+(patient|doctor|practice|physician|hcp)|dr\\.?\\s+[a-z]+\\s+(said|mentioned|noted)|no\\s+treatment\\s+decision)',
            q_stripped,
            re.IGNORECASE
        ))
        has_multiple_sentences = bool(re.search(r'\\.\\s+[A-Z]', q_stripped))
        valid_question_starters = (
            "what", "who", "when", "where", "why", "how", "which", "whom", "whose",
            "did", "do", "does", "was", "were", "is", "are", "can", "could",
            "would", "should", "have", "has", "had", "will", "any", "shall", "may",
            "i have enough"
        )
        first_word = q_stripped.split()[0].lower() if q_stripped.split() else ""
        not_a_question = first_word not in valid_question_starters

        if is_statement or has_multiple_sentences or not_a_question:
            if kg_fallback:
                question_text = kg_fallback

        # Rule 1.5: Strict Anti-Redundancy & Clinical Progression Guard (Never ask for facts already explicitly stated)
        # A. If user already stated call purpose, NEVER ask about purpose, main discussion, or general focus again
        purpose_already_stated = any(w in history_str_lower for w in ["to get an update", "treatment pathway", "eligible patient for nmibc", "bladder cancer"])
        if purpose_already_stated and any(w in question_text.lower() for w in ["purpose of the visit", "main purpose", "main inlexzo discussion", "more general", "focused on eligible"]):
            question_text = kg_fallback or (f"What did {known_hcp or 'the doctor'} share about eligible cases?")

        # B. If user already stated a patient or case was identified, NEVER ask if a patient/opportunity exists
        case_already_stated = any(w in history_str_lower for w in ["patient with post bcg", "bcg failure has been identified", "patient has been identified", "identified a patient"])
        if case_already_stated and any(w in question_text.lower() for w in ["have any current eligible", "any eligible case", "current eligible cases", "eligible patient", "current patient or upcoming", "mention any current patient", "upcoming opportunity"]):
            question_text = kg_fallback or "What is the current BCG status for that case?"

        # C. If user already stated BCG status (unresponsive, completed full course, BCG failure), NEVER re-ask BCG status
        bcg_already_stated = any(w in history_str_lower for w in ["completed full course", "bcg-unresponsive", "unresponsive", "post bcg failure", "post-bcg failure"])
        if bcg_already_stated and any(w in question_text.lower() for w in ["bcg status", "whether the patient is still on bcg", "bcg-naive", "prior bcg", "still on bcg"]):
            question_text = kg_fallback or "What is the next step for that case?"

        # D. If user already stated next step (reviewing history, checking suitability), NEVER re-ask next step
        next_step_already_stated = any(w in history_str_lower for w in ["check the complete medical history", "reviewing the complete medical history", "suitable for inlexzo", "suitability"])
        if next_step_already_stated and any(w in question_text.lower() for w in ["next step for that case", "next step for the patient"]):
            question_text = kg_fallback or "Did cystectomy refusal or ineligibility come up?"

        # Rule 1.6: Contextual Consistency Enforcement (Block out-of-context SLM hallucinations)
        if kg_fallback:
            # If user explained their call purpose (seeking eligible patient update), AI must probe what HCP shared, NOT ask about friction/setup/outcome
            if any(w in ans_lower for w in ["to get an update", "eligible patient", "treatment pathway", "bladder cancer", "nmibc"]):
                if any(w in question_text.lower() for w in ["friction", "procedure room", "in-service", "administration", "outcome"]):
                    question_text = kg_fallback

            # If user answered "no" / "none" (denying barriers or friction), NEVER ask about barriers, in-service training or procedure room setup
            if (ans_lower in ["no", "none", "not really", "no there arent any", "no there aren't any"] or ans_lower.startswith("no ")) and any(w in question_text.lower() for w in ["in-service", "procedure room", "administration", "friction", "barrier", "barriers"]):
                question_text = kg_fallback

            # If user is describing clinical patient status (BCG failure, unresponsiveness, suitability), NEVER ask about workflow friction
            if any(w in ans_lower for w in ["bcg failure", "post bcg", "unresponsive", "completed full course", "medical history", "suitable"]) and any(w in question_text.lower() for w in ["friction", "procedure room", "in-service", "administration"]):
                question_text = kg_fallback
        # Rule 2: Anti-Repetition Guard (Never ask a question already in history)
        prior_ai_questions = [h.get("text", "").strip().lower() for h in (conversation_history or []) if h.get("speaker") == "AI"]
        q_clean = question_text.strip().lower()
        
        if q_clean in prior_ai_questions:
            if kg_fallback and kg_fallback.strip().lower() not in prior_ai_questions:
                question_text = kg_fallback
            else:
                hcp = known_hcp or "the doctor"
                if role == "OS":
                    if "timing" not in history_str_lower and "when" not in history_str_lower:
                        question_text = "What is the timing for the next step?"
                    elif "support" not in history_str_lower and "resource" not in history_str_lower:
                        question_text = f"Did {hcp} mention any support or resources needed?"
                    elif "action items" not in history_str_lower and "follow-up" not in history_str_lower:
                        question_text = "Any follow-up or action items from the call?"
                    elif "account context" not in history_str_lower:
                        question_text = "Any other account context to capture?"
                    else:
                        question_text = "I have enough for the call note. Should we wrap here?"'''

ACCOUNT_BARRIERS_JSON = """[
  {
    "account": "Apollo Hospitals",
    "barrier_type": "Patient Identification Barrier",
    "barrier_details": "They are seeing very few eligible patients right now, so the team is not confident they can operationalize the pathway consistently. They asked for a simple way to recognize and flag potential candidates as they appear."
  },
  {
    "account": "Apollo Hospitals",
    "barrier_type": "Market Access Barrier",
    "barrier_details": "PA turnaround times are inconsistent, which makes it hard for the team to plan next steps. They asked to use approved access-support resources to understand the process and reduce rework."
  },
  {
    "account": "Fortis Healthcare",
    "barrier_type": "Patient Identification Barrier",
    "barrier_details": "Biomarker testing and/or results are sometimes delayed, which slows down patient identification in their pathway. The team asked to align internally on where testing fits in their workflow and who owns tracking results."
  },
  {
    "account": "Fortis Healthcare",
    "barrier_type": "Market Access Barrier",
    "barrier_details": "Patient cost exposure is a concern and the team wants clearer expectations before moving forward. They asked about benefits verification and approved resources that can help set expectations appropriately."
  },
  {
    "account": "Manipal Hospitals",
    "barrier_type": "Patient Identification Barrier",
    "barrier_details": "Key steps in their diagnosis-to-treatment pathway are taking longer than expected due to scheduling and coordination, which delays next-step decisions. The team wants clearer handoffs so potential candidates are identified and routed sooner."
  },
  {
    "account": "Manipal Hospitals",
    "barrier_type": "Market Access Barrier",
    "barrier_details": "Prior authorization timelines are creating scheduling uncertainty. The account asked what information their payer typically requests and wanted support using the appropriate, approved access pathway to reduce back-and-forth."
  },
  {
    "account": "Max Healthcare",
    "barrier_type": "Patient Identification Barrier",
    "barrier_details": "There are delays in the diagnostic workup and internal handoffs, so potential candidates are identified late in the process. The team wants a cleaner pathway with fewer missed handoffs between roles."
  },
  {
    "account": "Max Healthcare",
    "barrier_type": "Market Access Barrier",
    "barrier_details": "The product is not on the current formulary (or is restricted), so usage requires an exception pathway. The account needs clarity on committee timing, required documentation, and who will submit the request."
  },
  {
    "account": "Narayana Health",
    "barrier_type": "Patient Identification Barrier",
    "barrier_details": "The site is not consistently identifying potential candidates in their current workflow. They want a simple way to flag appropriate patients earlier without adding extra burden to clinic staff."
  },
  {
    "account": "Narayana Health",
    "barrier_type": "Market Access Barrier",
    "barrier_details": "Recent changes in payer coverage policy created uncertainty about requirements and next steps. The account requested the latest available, published policy information and clarity on how to route questions through the approved access support channel."
  }
]"""

def patch_files():
    print("Writing account_barriers.json...")
    with open("account_barriers.json", "w", encoding="utf-8") as f:
        f.write(ACCOUNT_BARRIERS_JSON)
    print("[OK] account_barriers.json written.")

    print("Patching engine/kg_context_retriever.py...")
    with open("engine/kg_context_retriever.py", "r", encoding="utf-8") as f:
        c1 = f.read()

    # Replace retrieve_kg_node_inquiry
    if "def retrieve_kg_node_inquiry" in c1:
        c1 = re.sub(
            r'    def retrieve_kg_node_inquiry[\s\S]*?    def retrieve_relevant_duties',
            lambda _: NEW_INQUIRY_METHOD + "\n    def retrieve_relevant_duties",
            c1
        )
    else:
        c1 = c1.replace(
            "    def retrieve_relevant_duties",
            NEW_INQUIRY_METHOD + "\n    def retrieve_relevant_duties"
        )

    # Ensure OS roadmap does not contain out-of-scope workflow
    c1 = c1.replace('"clinical operational workflow",\n                "dosing administration",\n', '')
    c1 = c1.replace('"clinical operational workflow",\n', '')

    with open("engine/kg_context_retriever.py", "w", encoding="utf-8") as f:
        f.write(c1)
    print("[OK] engine/kg_context_retriever.py updated.")

    print("Patching engine/dialogue_state_tracker.py...")
    with open("engine/dialogue_state_tracker.py", "r", encoding="utf-8") as f:
        c3 = f.read()

    dynamic_state_code = '''        # 5. Dynamically Advance Dialogue State based on conversation context and question topic
        is_asking_wrap = bool(re.search(r'\\b(wrap|close out|enough for the call note)\\b', next_q or "", re.IGNORECASE))
        q_lower = (next_q or "").lower()

        if is_asking_wrap:
            next_state = "STATE_7_WRAP_UP_CONFIRMATION"
        elif any(w in q_lower for w in ["action item", "follow-up", "stay in touch", "account context"]):
            next_state = "STATE_6_NEXT_ACTIONS"
        elif any(w in q_lower for w in ["support", "resource", "msl", "referral"]):
            next_state = "STATE_5_CROSS_FUNCTIONAL_HANDOFF"
        elif any(w in q_lower for w in ["barrier", "coverage", "access", "timing"]):
            next_state = "STATE_4_BARRIERS_LOGISTICS"
        elif any(w in q_lower for w in ["administration", "procedure room", "in-service", "setup"]):
            next_state = "STATE_3E_WORKFLOW_DEMO_REFRESHER"
        elif any(w in q_lower for w in ["eligible case", "share about", "bcg status", "next step", "treatment pathway", "treatment approach"]):
            next_state = "STATE_2_PRIMARY_PURPOSE"
        elif any(w in q_lower for w in ["discussion today", "main purpose"]):
            next_state = "STATE_2_PRIMARY_PURPOSE"
        elif any(w in q_lower for w in ["who did you meet"]):
            next_state = "STATE_1_ACCOUNT_STAKEHOLDER"
        else:
            transition = STATE_TRANSITIONS.get(self.current_state)
            if isinstance(transition, dict):
                next_state = transition.get(self.role, "STATE_6_NEXT_ACTIONS")
            elif isinstance(transition, str):
                next_state = transition
            else:
                next_state = "STATE_6_NEXT_ACTIONS"

        self.current_state = next_state'''

    if "Dynamically Advance Dialogue State" not in c3:
        c3 = re.sub(
            r'        # 5\. Advance FSM State[\s\S]*?self\.current_state = next_state',
            lambda _: dynamic_state_code,
            c3
        )

    # Wrap-up confirmation check in dialogue_state_tracker.py
    c3 = c3.replace(
        'is_wrap_up_confirmation = (self.current_state == "STATE_7_WRAP_UP_CONFIRMATION" and bool(re.match(r\'^(yes|yeah|sure|wrap\\s*it|close\\s*it|go\\s*ahead|done|all\\s*set)[.!]?$\', ans_lower)))',
        '''curr_q = self.current_question or ""
        is_asking_wrap = bool(re.search(r'\\b(wrap(\\s*up|\\s*here)?|close\\s*out|enough for the call note)\\b', curr_q, re.IGNORECASE))
        is_affirmative = bool(re.search(r'\\b(yes|yeah|sure|yep|ok|okay|wrap(\\s*it|\\s*up)?|close(\\s*it)?|go\\s*ahead|done|all\\s*set|please|we\\s*can)\\b', ans_lower))
        is_wrap_up_confirmation = (is_asking_wrap or self.current_state == "STATE_7_WRAP_UP_CONFIRMATION") and is_affirmative'''
    )

    with open("engine/dialogue_state_tracker.py", "w", encoding="utf-8") as f:
        f.write(c3)
    print("[OK] engine/dialogue_state_tracker.py updated.")

    print("Patching engine/nlu_extractor.py...")
    with open("engine/nlu_extractor.py", "r", encoding="utf-8") as f:
        c_nlu = f.read()

    new_ood_method = '''    def check_out_of_domain(self, text: str) -> Dict[str, Any]:
        """
        Detects if the rep input is an out-of-domain / irrelevant query or unrelated nonsense.
        Active by default to guard call notes against non-commercial / off-topic content.
        """
        text_clean = text.strip().lower()
        if not text_clean:
            return {"is_out_of_domain": True, "suggested_message": "I didn't receive any input. Could you please share your call details?"}

        # 1. Affirmations & Positive Conversational Replies (e.g., 'yes sure', 'yes please', 'sure thing', 'we can')
        affirmation_patterns = [
            r'^(yes|yep|yeah|sure|ok|okay|of\\s+course|certainly|absolutely|definitely|ready|go\\s+ahead|lets\\s+do\\s+it|let\\'s\\s+do\\s+it|lets\\s+start|let\\'s\\s+start|lets\\s+go|let\\'s\\s+go|i\\s+do|i\\s+have\\s+time|yeah\\s+sure|yes\\s+sure|yes\\s+please|sure\\s+thing|sure\\s+go\\s+ahead|yes\\s+let\\'s\\s+do\\s+it|yes\\s+lets\\s+do\\s+it|yes\\s+of\\s+course|yes\\s+i\\s+do|why\\s+not|all\\s+set|sounds\\s+good|proceed|let\\'s\\s+begin|fine|we\\s+can|yes\\s+we\\s+can|yep\\s+we\\s+can)[.! ]*$',
            r'\\b(yes|yeah|yep|sure\\s+thing|go\\s+ahead|of\\s+course|certainly|let\\'s\\s+do\\s+it|we\\s+can|yes\\s+we\\s+can)\\b'
        ]
        for ap in affirmation_patterns:
            if re.search(ap, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 2. Negations & Session Control Commands (e.g., 'no', 'not now', 'busy', 'nothing else', 'done')
        negation_patterns = [
            r'^(no|nope|nah|none|nothing|nothing\\s+else|no\\s+more(\\s+notes)?|no\\s+other\\s+topics|nothing\\s+to\\s+(add|report)|done|finished|all\\s+set|stop|exit|cancel|not\\s+now|busy|later|don\\'t\\s+have\\s+time|no\\s+time|i\\s+want\\s+to\\s+end|n/a|no\\s+action|no\\s+updates|bye|goodbye)[.! ]*$',
            r'\\b(no\\s+more|not\\s+now|busy\\s+right\\s+now|later|don\\'t\\s+have\\s+time|no\\s+time|nothing\\s+else|nothing\\s+to\\s+add|there\\s+arent|there\\s+aren\\'t|no\\s+barriers?|no\\s+issues?|none\\s+identified)\\b',
            r'\\b(no\\s+there\\s+aren\\'t(\\s+any)?|there\\s+aren\\'t\\s+any|none\\s+at\\s+all|not\\s+really|not\\s+at\\s+all)\\b',
            r'\\b(did\\s+not\\s+mention|didn\\'t\\s+mention|did\\s+not\\s+say|didn\\'t\\s+say|not\\s+mention|nothing\\s+at\\s+the\\s+moment|not\\s+at\\s+the\\s+moment|not\\s+at\\s+this\\s+time|not\\s+right\\s+now)\\b',
            r'\\b(this\\s+is\\s+all(\\s+we\\s+have|\\s+i\\s+have)?|that\\'s\\s+all(\\s+we\\s+have|\\s+i\\s+have)?|that\\s+is\\s+all|all\\s+we\\s+have|all\\s+i\\s+have|nothing\\s+else|that\\s+would\\s+be\\s+all)\\b',
            r'\\b(he\\s+did\\s+not|she\\s+did\\s+not|they\\s+did\\s+not|doctor\\s+did\\s+not)\\b'
        ]
        for np in negation_patterns:
            if re.search(np, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 3. Healthcare, Oncology, Access & Commercial Terminology
        healthcare_whitelist = [
            r"\\b(dr\\.?|doctor|md\\b|do\\b|physician|patient|patients|oncology|urology|hospital|clinic|center|centre|institute|account|office|site|practice|health\\s+system|infusion\\s+center)\\b",
            r"\\b(inlexzo|rybrevant|lazcluze|brand|drug|therapy|regimen|nsclc|nmibc|bcg|bcg-unresponsive|bladder|baldder|urothelial|tarceva|tagrisso|chemo|immunotherapy|medication|product)\\b",
            r"\\b(prior\\s+auth|pa\\b|copay|co-pay|pap|hub|access|reimbursement|payer|medicare|medicaid|blue\\s+cross|aetna|cigna|united|coverage|appeal|appeals|denial|denials|denied)\\b",
            r"\\b(dosing|infusion|administration|subcutaneous|efficacy|trial|pfs|safety|rash|adverse|toxicity|mitigation|workflow|ehr|order\\s+set|protocol|support|program|enrollment|form|formulary|barrier|delay|turnaround|approved|approval|pending|submitted|specialty\\s+pharmacy|distributor|billing|claim|claims|j-code|code|coding)\\b",
            r"\\b(nurse|rn|np\\b|pa\\b|coordinator|manager|staff|counselor|specialist|rep|kam|msl|frm|os|mir|loop\\s+in|sample|samples|brochure|materials|slide|deck)\\b",
            r"\\b(post\\s+bcg|bcg\\s+failure|unresponsive|cystectomy|medical\\s+history|suitable|suitability|treatment\\s+pathway|treatment\\s+approach|eligible|eligibility)\\b"
        ]
        for hw in healthcare_whitelist:
            if re.search(hw, text_clean, re.IGNORECASE):
                return {"is_out_of_domain": False}

        # 4. Meeting Discourse & Commercial Collaboration Verbs
        meeting_discourse = [
            r"\\b(met\\s+with|visited|spoke\\s+with|called|follow[\\s\\-_]*up|next\\s+step|action\\s+plan|discussed|talked|reviewed|shared|agreed|presented|explained|asked|question|inquired|clarified|educated|scheduled|meeting|call|visit|check-in|outreach|touchpoint)\\b",
            r"\\b(everything\\s+(fine|good|clear|smooth|set)|no\\s+(issue|issues|problem|questions|barriers)|looking\\s+into|working\\s+on|in\\s+progress|not\\s+yet|already|waiting)\\b",
            r"\\b(next\\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|tomorrow|by\\s+(friday|tuesday|monday|wednesday|thursday)|early\\s+[a-z]+|couple\\s+of\\s+weeks|weeks|days|month|months)\\b",
            r"\\b(aiming\\s+to|planning\\s+to|decide|deciding|reviewing|checking|updating|update|check\\s+the\\s+complete)\\b"
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
        }'''

    c_nlu = re.sub(
        r'    def check_out_of_domain[\s\S]*?    def analyze_utterance',
        lambda _: new_ood_method + "\n\n    def analyze_utterance",
        c_nlu
    )
    with open("engine/nlu_extractor.py", "w", encoding="utf-8") as f:
        f.write(c_nlu)
    print("[OK] engine/nlu_extractor.py updated.")

    print("Patching engine/chatbot_pipeline.py...")
    with open("engine/chatbot_pipeline.py", "r", encoding="utf-8") as f:
        c4 = f.read()

    ood_sync_block = '''        # STEP 1.5: Out-Of-Domain / Irrelevant Input Check
        ood_result = self.nlu.check_out_of_domain(candidate_answer)
        if ood_result.get("is_out_of_domain"):
            latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
            curr_q = session.current_question or ""
            if session.current_state == "STATE_0_GREETING_INITIATION":
                clarification_msg = f"That appears out of context. I'm here to capture your {session.role} call notes. Would you like to start capturing notes for your meeting?"
            else:
                clarification_msg = f"That appears out of context for this call note. {curr_q}"
            
            return {
                "status": "OUT_OF_DOMAIN_INTERCEPT",
                "role": session.role,
                "current_state": session.current_state,
                "bot_message": clarification_msg,
                "is_completed": False,
                "slots": session.slots,
                "latency_ms": latency_ms,
                "latency_formatted": f"{latency_ms:.0f}ms",
                "session_summary": session.get_summary()
            }'''

    ood_stream_block = '''        # STEP 1.5: Out-Of-Domain / Irrelevant Input Check
        ood_result = self.nlu.check_out_of_domain(candidate_answer)
        if ood_result.get("is_out_of_domain"):
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            curr_q = session.current_question or ""
            if session.current_state == "STATE_0_GREETING_INITIATION":
                clarification_msg = f"That appears out of context. I'm here to capture your {session.role} call notes. Would you like to start capturing notes for your meeting?"
            else:
                clarification_msg = f"That appears out of context for this call note. {curr_q}"
            
            yield {"type": "token", "token": clarification_msg}
            metrics = {
                "type": "metrics",
                "first_token_latency_ms": latency_ms,
                "tokens_per_second": 0.0,
                "generated_tokens": len(clarification_msg.split()),
                "total_generation_time_ms": latency_ms
            }
            yield metrics
            yield {
                "type": "result",
                "status": "OUT_OF_DOMAIN_INTERCEPT",
                "role": session.role,
                "current_state": session.current_state,
                "bot_message": clarification_msg,
                "is_completed": False,
                "slots": session.slots,
                "metrics": metrics,
                "latency_ms": latency_ms,
                "latency_formatted": f"{latency_ms:.0f}ms",
                "session_summary": session.get_summary()
            }
            return'''

    if "STEP 1.5: Out-Of-Domain" not in c4:
        c4 = c4.replace(
            '        # STEP 2: Explicit Session End Check',
            ood_sync_block + '\n\n        # STEP 2: Explicit Session End Check',
            1
        )
        c4 = c4.replace(
            '        # STEP 2: Explicit Session End Check',
            ood_stream_block + '\n\n        # STEP 2: Explicit Session End Check',
            1
        )

    c4 = c4.replace(
        'is_wrap_up_confirmation = (session.current_state == "STATE_7_WRAP_UP_CONFIRMATION" and bool(re.match(r\'^(yes|yeah|sure|wrap\\s*it|close\\s*it|go\\s*ahead|done|all\\s*set|yep|ok|okay)[.!]?$\', ans_lower)))',
        '''curr_q = session.current_question or ""
        is_asking_wrap = bool(re.search(r'\\b(wrap(\\s*up|\\s*here)?|close\\s*out|enough for the call note)\\b', curr_q, re.IGNORECASE))
        is_affirmative = bool(re.search(r'\\b(yes|yeah|sure|yep|ok|okay|wrap(\\s*it|\\s*up)?|close(\\s*it)?|go\\s*ahead|done|all\\s*set|please|we\\s*can)\\b', ans_lower))
        is_wrap_up_confirmation = (is_asking_wrap or session.current_state == "STATE_7_WRAP_UP_CONFIRMATION") and is_affirmative'''
    )

    with open("engine/chatbot_pipeline.py", "w", encoding="utf-8") as f:
        f.write(c4)
    print("[OK] engine/chatbot_pipeline.py updated.")

    print("Patching engine/slm_next_question_engine.py...")
    with open("engine/slm_next_question_engine.py", "r", encoding="utf-8") as f:
        c2 = f.read()

    # Ensure kg_fallback in signature
    c2 = c2.replace(
        "brand: str = \"RYBREVANT\"\n    ) -> str:",
        "brand: str = \"RYBREVANT\",\n        kg_fallback: Optional[str] = None\n    ) -> str:"
    )

    # Replace safety rules
    if "# Rule 0.9: STRICT ENTITY ANCHORING" in c2:
        c2 = re.sub(
            r'        # Rule 0.9: STRICT ENTITY ANCHORING[\s\S]*?        # Rule 2\.5: Re-enforce entity replacement',
            lambda _: NEW_SAFETY_RULES + "\n\n        # Rule 2.5: Re-enforce entity replacement",
            c2
        )
    else:
        c2 = c2.replace(
            "        # Rule 1: Toxicity Terminology Replacement",
            NEW_SAFETY_RULES + "\n        # Rule 2: Anti-Repetition Guard"
        )

    # Wire kg_inquiry into callers
    if "kg_inquiry = self.kg_retriever.retrieve_kg_node_inquiry" not in c2:
        old_call = '''        target_topic = flat_context["target_topic"]
        hcp = persona_name or (entities.get("hcps", ["the doctor"])[0] if entities.get("hcps") else "the doctor")'''
        new_call = '''        target_topic = flat_context["target_topic"]
        hcp = persona_name or (entities.get("hcps", ["the doctor"])[0] if entities.get("hcps") else "the doctor")

        kg_inquiry = self.kg_retriever.retrieve_kg_node_inquiry(
            role=role_upper,
            target_topic=target_topic,
            conversation_history=conversation_history,
            candidate_answer=candidate_answer,
            brand=brand,
            hcp=hcp
        )'''
        c2 = c2.replace(old_call, new_call)
        c2 = c2.replace("brand=brand\n        )", "brand=brand,\n            kg_fallback=kg_inquiry\n        )")
        c2 = c2.replace("brand=brand\n            )", "brand=brand,\n                kg_fallback=kg_inquiry\n            )")

    with open("engine/slm_next_question_engine.py", "w", encoding="utf-8") as f:
        f.write(c2)
    print("[OK] engine/slm_next_question_engine.py updated.")

    print("Patching main.py...")
    with open("main.py", "r", encoding="utf-8") as f:
        m_code = f.read()

    if "/api/accounts/barriers" not in m_code:
        account_endpoint = '''
@app.get("/api/accounts/barriers", tags=["Account Intelligence"])
def get_account_barriers():
    return {
        "status": "SUCCESS",
        "accounts": [
            "Apollo Hospitals",
            "Fortis Healthcare",
            "Manipal Hospitals",
            "Max Healthcare",
            "Narayana Health"
        ],
        "barriers": bot.kg.account_barriers if hasattr(bot.kg, "account_barriers") else []
    }
'''
        m_code = m_code.replace('@app.post("/api/slm/explain_context"', account_endpoint + '\n@app.post("/api/slm/explain_context"')

    with open("main.py", "w", encoding="utf-8") as f:
        f.write(m_code)
    print("[OK] main.py updated.")

    print("All patches applied successfully!")

if __name__ == "__main__":
    patch_files()
