"""
Dialogue State Tracker (DST)
=============================
Maintains conversational context, slot values, turn progression, and
state tracking across multi-turn oncology call note interviews.
"""

import re
from typing import Dict, List, Any, Optional

STATE_TRANSITIONS = {
    "STATE_0_GREETING_INITIATION": "STATE_1_ACCOUNT_STAKEHOLDER",
    "STATE_1_ACCOUNT_STAKEHOLDER": "STATE_2_PRIMARY_PURPOSE",
    "STATE_2_PRIMARY_PURPOSE": {
        "FRM": "STATE_3A_PRIOR_AUTH_PAYER",
        "OS": "STATE_3E_WORKFLOW_DEMO_REFRESHER"
    },
    "STATE_3A_PRIOR_AUTH_PAYER": "STATE_3B_AFFORDABILITY_COPAY_PAP",
    "STATE_3B_AFFORDABILITY_COPAY_PAP": "STATE_3C_HUB_SPECIALTY_PHARMACY",
    "STATE_3C_HUB_SPECIALTY_PHARMACY": "STATE_6_NEXT_ACTIONS",
    "STATE_3E_WORKFLOW_DEMO_REFRESHER": "STATE_4_BARRIERS_LOGISTICS",
    "STATE_4_BARRIERS_LOGISTICS": "STATE_5_CROSS_FUNCTIONAL_HANDOFF",
    "STATE_5_CROSS_FUNCTIONAL_HANDOFF": "STATE_6_NEXT_ACTIONS",
    "STATE_6_NEXT_ACTIONS": "STATE_7_WRAP_UP_CONFIRMATION",
    "STATE_7_WRAP_UP_CONFIRMATION": "COMPLETED"
}

__all__ = ["DialogueSession", "DialogueStateTracker", "STATE_TRANSITIONS"]

class DialogueSession:
    def __init__(self, role: str, session_id: str = "SESSION_001"):
        self.session_id = session_id
        self.role = role.upper()
        self.turn_index = 0
        self.current_state = "STATE_0_GREETING_INITIATION"
        self.current_question = (
            f"Hello, do you have a minute to capture the {self.role} call notes?"
        )
        self.slots = {
            "user_name": None,
            "brand": None,
            "hcp_name": None,
            "account_name": None,
            "stakeholders": [],
            "payer_status": None,
            "barriers": [],
            "account_barrier": None,
            "barrier_case": None,
            "barrier_detected": False,
            "barrier_resolved": False,
            "live_graph": None,
            "graph_mermaid": None,
            "next_action": None,
            "action_owner": None,
            "target_timing": None
        }
        self.covered_topics = set()
        self.history = [
            {"speaker": "AI", "text": self.current_question, "state": self.current_state}
        ]
        self.is_completed = False

    def update_state(self, candidate_answer: str, nlu_result: Dict[str, Any], next_q: str, slm_result: Optional[Dict[str, Any]] = None):
        """Updates slot values and advances the conversational state."""
        self.turn_index += 1
        ans_clean = candidate_answer.strip()
        ans_lower = ans_clean.lower()
        
        # 1. Update Slots from Extracted Entities
        entities = nlu_result.get("entities", {})
        if entities.get("brands") and not self.slots["brand"]:
            self.slots["brand"] = entities["brands"][0]
        if entities.get("hcps") and not self.slots["hcp_name"]:
            self.slots["hcp_name"] = entities["hcps"][0]
        if entities.get("accounts") and not self.slots["account_name"]:
            self.slots["account_name"] = entities["accounts"][0]
        if entities.get("stakeholder_roles"):
            for s in entities["stakeholder_roles"]:
                if s not in self.slots["stakeholders"]:
                    self.slots["stakeholders"].append(s)
        if entities.get("dates_or_timings") and not self.slots["target_timing"]:
            self.slots["target_timing"] = entities["dates_or_timings"][0]
            
        if slm_result:
            if slm_result.get("account_barrier"):
                self.slots["account_barrier"] = slm_result["account_barrier"]
            if slm_result.get("barrier_case"):
                self.slots["barrier_case"] = slm_result["barrier_case"]
                self.slots["barrier_detected"] = True
            if slm_result.get("live_graph"):
                self.slots["live_graph"] = slm_result["live_graph"]
            if slm_result.get("graph_mermaid"):
                self.slots["graph_mermaid"] = slm_result["graph_mermaid"]
            
        # 2. Record Covered Topics
        for top in nlu_result.get("topics", []):
            self.covered_topics.add(top)

        # 3. Record History
        self.history.append({
            "speaker": "User",
            "text": candidate_answer,
            "turn_index": self.turn_index,
            "state": self.current_state
        })
        
        curr_q = self.current_question or ""
        is_asking_wrap = bool(re.search(r'\b(wrap(\s*up|\s*here)?|close\s*out|enough for the call note)\b', curr_q, re.IGNORECASE))
        is_greeting_refusal = (self.current_state == "STATE_0_GREETING_INITIATION" and bool(re.match(r'^(no|not now|busy|don\'?t have time|no time|later)[.!]?$', ans_lower)))
        is_explicit_stop = bool(re.search(r'\b(close\s*(out)?\s*(the|this)?\s*call\s*note|close\s*note|end\s*(the)?\s*(call|session|interview)|stop\s*session|quit\s*note|wrap\s*(it|up|this)|please\s*wrap|wrap\s*up)\b', ans_lower))
        is_affirmative = bool(re.search(r'\b(yes|yeah|sure|yep|ok|okay|wrap(\s*it|\s*up)?|close(\s*it)?|go\s*ahead|done|all\s*set|please|we\s*can)\b', ans_lower))
        is_wrap_up_confirmation = (is_asking_wrap or self.current_state == "STATE_7_WRAP_UP_CONFIRMATION") and is_affirmative

        if is_greeting_refusal:
            self.current_state = "COMPLETED"
            self.is_completed = True
            closing_msg = "Understood, no problem. Please start a new session whenever you are ready to capture notes."
            self.current_question = closing_msg
            self.history.append({
                "speaker": "AI",
                "text": closing_msg,
                "turn_index": self.turn_index,
                "state": "COMPLETED"
            })
            return

        if is_explicit_stop or is_wrap_up_confirmation:
            self.current_state = "COMPLETED"
            self.is_completed = True
            hcp_disp = self.slots.get('hcp_name') or 'this visit'
            closing_msg = f"Thank you, the call notes for {hcp_disp} have been captured and logged compliantly. Session closed."
            self.current_question = closing_msg
            self.history.append({
                "speaker": "AI",
                "text": closing_msg,
                "turn_index": self.turn_index,
                "state": "COMPLETED"
            })
            return

        # 5. Dynamically Advance Dialogue State based on conversation context and question topic
        is_asking_wrap = bool(re.search(r'\b(wrap|close out|enough for the call note)\b', next_q or "", re.IGNORECASE))
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

        self.current_state = next_state
        self.current_question = next_q
        self.is_completed = False
        self.history.append({
            "speaker": "AI",
            "text": next_q,
            "turn_index": self.turn_index,
            "state": self.current_state
        })
            
    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "turn_count": self.turn_index,
            "current_state": self.current_state,
            "is_completed": self.is_completed,
            "slots": self.slots,
            "covered_topics": list(self.covered_topics)
        }

    def get_cumulative_transcript(self) -> Optional[str]:
        """
        Returns the full conversation history as a cumulative transcript
        string in the format: "AI: ...\nUser: ...\nAI: ..."
        Returns None if no history exists (cold-start / turn 0).
        """
        if not self.history:
            return None
        lines = []
        for entry in self.history:
            speaker = entry.get("speaker", "User")
            text = entry.get("text", "")
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

DialogueStateTracker = DialogueSession
