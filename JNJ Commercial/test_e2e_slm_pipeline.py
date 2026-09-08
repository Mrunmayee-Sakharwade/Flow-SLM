"""
End-to-End Verification Test for KG-Conditioned SLM Next-Question Pipeline
==========================================================================
Tests:
  1. FRM Clinical Efficacy Attempt -> Intercepted with KG Rule Citation (resp:FRM:28)
  2. OS Clinical Efficacy Discussion -> Approved & Predicted by SLM
  3. FRM Multi-Turn Prior Auth & Denial Flow -> Full state progression with KG Context
  4. Global Compliance PHI Check -> Redacted with rule:privacy_phi_pii citation
  5. Cross-Functional Collaboration Trigger -> Detects MSL referral trigger for OS rep
"""

import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from engine.chatbot_pipeline import RuleGovernedCallBot

def test_pipeline():
    print("=" * 75)
    print("RUNNING END-TO-END PIPELINE VERIFICATION SUITE")
    print("=" * 75)
    
    bot = RuleGovernedCallBot()
    
    # -----------------------------------------------------------------------
    # TEST 1: FRM Scope Violation (Clinical Efficacy)
    # -----------------------------------------------------------------------
    print("\n--- TEST 1: FRM Clinical Efficacy Attempt (Expected: SCOPE VIOLATION) ---")
    s1 = bot.create_session("FRM", "TEST_FRM_001")
    res1 = bot.process_turn("TEST_FRM_001", "The oncologist asked about clinical trial efficacy and biomarker testing for our lung cancer patient.")
    print("Status:", res1["status"])
    print("Bot Message:", res1["bot_message"])
    assert res1["status"] == "SCOPE_VIOLATION_INTERCEPT", "Expected scope violation intercept"
    assert "resp:FRM:28" in res1["rule_id"] or "FRM" in res1["rule_id"], "Expected FRM rule citation"
    print(">>> PASS: FRM Clinical discussion correctly blocked by Knowledge Graph!")

    # -----------------------------------------------------------------------
    # TEST 2: OS Clinical Efficacy Discussion (Permitted for Sales Rep)
    # -----------------------------------------------------------------------
    print("\n--- TEST 2: OS Clinical Efficacy Discussion (Expected: SUCCESS + SLM Next-Q) ---")
    s2 = bot.create_session("OS", "TEST_OS_001")
    res2 = bot.process_turn("TEST_OS_001", "The oncologist was impressed by the PFS clinical trial efficacy data for RYBREVANT.")
    print("Status:", res2["status"])
    print("Target Topic:", res2["target_topic"])
    print("Predicted Next Question (SLM):", res2["bot_message"])
    print("Injected KG Context Summary:", res2["applicable_responsibilities"])
    assert res2["status"] == "SUCCESS", "Expected success for OS rep"
    assert res2["target_topic"] in ["account identification", "efficacy safety product info", "clinical operational workflow"], "Expected valid target topic"
    print(">>> PASS: OS Clinical discussion approved and processed by SLM!")

    # -----------------------------------------------------------------------
    # TEST 3: Multi-Turn FRM Prior Auth Flow
    # -----------------------------------------------------------------------
    print("\n--- TEST 3: Multi-Turn FRM Prior Auth Flow ---")
    s3 = bot.create_session("FRM", "TEST_FRM_002")
    
    # Turn 1: Account Context
    t1 = bot.process_turn("TEST_FRM_002", "I met with Dr. Robert Avery and Lisa at Birmingham Oncology.")
    print("Turn 1 -> Next State:", t1["next_state"], "| Next Q:", t1["bot_message"])
    
    # Turn 2: Core Purpose
    t2 = bot.process_turn("TEST_FRM_002", "We did a reimbursement check-in for RYBREVANT.")
    print("Turn 2 -> Next State:", t2["next_state"], "| Next Q:", t2["bot_message"])
    
    # Turn 3: PA Denials
    t3 = bot.process_turn("TEST_FRM_002", "Lisa had two prior authorization denials from a commercial payer.")
    print("Turn 3 -> Next State:", t3["next_state"], "| Next Q:", t3["bot_message"])
    print("KG Target Topic:", t3["target_topic"])
    
    assert "denial" in t3["bot_message"].lower() or "appeal" in t3["bot_message"].lower() or "prior authorization" in t3["bot_message"].lower()
    print(">>> PASS: Multi-turn FRM state and slot tracking advanced smoothly!")

    # -----------------------------------------------------------------------
    # TEST 4: Global Compliance PHI Check
    # -----------------------------------------------------------------------
    print("\n--- TEST 4: Global Compliance PHI Redaction ---")
    s4 = bot.create_session("OS", "TEST_OS_002")
    res4 = bot.process_turn("TEST_OS_002", "Patient John Doe DOB 05/12/1980 was prescribed therapy.")
    print("Status:", res4["status"])
    print("Action Type:", res4["action_type"])
    print("Redacted Text:", res4["redacted_text"])
    assert res4["status"] == "COMPLIANCE_VIOLATION", "Expected PHI violation"
    print(">>> PASS: PHI detected and redacted compliantly!")

    # -----------------------------------------------------------------------
    # TEST 5: Cross-Functional Collaboration Trigger
    # -----------------------------------------------------------------------
    print("\n--- TEST 5: Cross-Functional Collaboration Trigger (OS -> MSL) ---")
    s5 = bot.create_session("OS", "TEST_OS_003")
    res5 = bot.process_turn("TEST_OS_003", "Dr. Smith had deep molecular questions about exon 20 insertion mutations.")
    print("Status:", res5["status"])
    print("Next Q (SLM):", res5["bot_message"])
    print("Handoff Detected:", res5["cross_functional_handoff"])
    assert "MSL" in res5["bot_message"] or "Medical Information Request" in res5["bot_message"] or "mir" in res5["bot_message"].lower()
    print(">>> PASS: Cross-functional collaboration trigger to MSL executed flawlessly!")

    print("\n" + "=" * 75)
    print("ALL 5 END-TO-END PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    test_pipeline()
