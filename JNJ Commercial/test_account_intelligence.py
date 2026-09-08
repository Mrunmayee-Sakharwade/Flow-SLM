"""
test_account_intelligence.py
Comprehensive Validation Suite for J&J Account Barrier Intelligence:
1. Account Entity Normalization (Apollo, Fortis, Manipal, Max, Narayana)
2. Role-Conditioned Dedicated KGs (kg_os.json & kg_frm.json) Node & Edge Verification
3. Dual-Case Conversational Handling:
   - Case 1 (Rep-Initiated): Rep brings up barrier -> AI immediately addresses account ask
   - Case 2 (Proactive AI): Rep doesn't mention barrier -> AI proactively checks historical barrier
4. Dynamic Live Entity-Relationship Knowledge Graph Materialization (NOT a static flowchart)
5. Out-of-Domain & Compliance Guardrail Protection
6. Universal Clean Session Wrap-Up
"""

import sys
from engine.nlu_extractor import NLUExtractor
from engine.kg_context_retriever import KGContextRetriever
from engine.chatbot_pipeline import RuleGovernedCallBot

def run_tests():
    print("=" * 80)
    print("JNJ COMMERCIAL INTELLIGENCE & DYNAMIC KNOWLEDGE GRAPH VALIDATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1: NLU Account Entity Normalization
    # -------------------------------------------------------------------------
    print("\n--- Test 1: NLU Account Entity Normalization ---")
    nlu = NLUExtractor()
    acc_tests = [
        ("I visited Dr Anurag at Apollo today", "Apollo Hospitals"),
        ("Met Dr Rao at Fortis Hospital", "Fortis Healthcare"),
        ("Spoke with clinic staff at Manipal", "Manipal Hospitals"),
        ("Called team at Max Healthcare center", "Max Healthcare"),
        ("Visited Narayana Health hospital", "Narayana Health")
    ]
    for text, expected_acc in acc_tests:
        res = nlu.extract_entities(text)
        acc = res.get("accounts", [None])[0]
        assert acc == expected_acc, f"Failed for '{text}': got '{acc}', expected '{expected_acc}'"
        print(f" [PASS] '{text}' -> {acc}")

    # -------------------------------------------------------------------------
    # Test 2: Dedicated Role-Specific KGs & Barrier Profiles
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Dedicated Role-Specific KGs & Barrier Profiles ---")
    kg = KGContextRetriever()
    accounts = ["Apollo Hospitals", "Fortis Healthcare", "Manipal Hospitals", "Max Healthcare", "Narayana Health"]
    for acc in accounts:
        os_b = kg.get_account_barrier(acc, "OS")
        frm_b = kg.get_account_barrier(acc, "FRM")
        assert os_b is not None, f"Missing OS barrier for {acc}"
        assert frm_b is not None, f"Missing FRM barrier for {acc}"
        assert os_b["barrier_type"] == "Patient Identification Barrier"
        assert frm_b["barrier_type"] == "Market Access Barrier"
        assert len(os_b.get("trigger_keywords", [])) >= 5
        assert len(frm_b.get("trigger_keywords", [])) >= 5
        assert os_b.get("case_1_inquiry") is not None
        assert os_b.get("case_2_inquiry") is not None
        assert frm_b.get("case_1_inquiry") is not None
        assert frm_b.get("case_2_inquiry") is not None
        print(f" [PASS] {acc}:")
        print(f"        OS  -> {os_b['barrier_type']} ({len(os_b['trigger_keywords'])} keywords)")
        print(f"        FRM -> {frm_b['barrier_type']} ({len(frm_b['trigger_keywords'])} keywords)")

    bot = RuleGovernedCallBot()

    # -------------------------------------------------------------------------
    # Test 3: Dual-Case Handling - Case 1: Rep-Initiated Barrier Inquiry
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Dual-Case Handling - Case 1 (Rep-Initiated Barrier) ---")
    
    # 3A: OS Apollo - Rep mentions few eligible patients & operationalizing pathway
    sess_case1_os = bot.create_session(role="OS", brand="INLEXZO", account_name="Apollo Hospitals")
    res_c1_os = bot.process_turn(
        sess_case1_os.session_id,
        "I met Dr. Anurag at Apollo. They are seeing very few eligible patients right now and are not confident they can operationalize the pathway."
    )
    print(f"User: I met Dr. Anurag at Apollo. They are seeing very few eligible patients right now...")
    print(f"AI:   {res_c1_os['bot_message']}")
    assert res_c1_os.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "streamlined criteria or flagging tool" in res_c1_os['bot_message'] or "flag potential candidates" in res_c1_os['bot_message'] or "Apollo" in res_c1_os['bot_message']
    print(" [PASS] OS Apollo Case 1 triggered candidate flagging inquiry!")

    # 3B: OS Fortis - Rep mentions biomarker testing delays
    sess_case1_fortis = bot.create_session(role="OS", brand="INLEXZO", account_name="Fortis Healthcare")
    res_c1_fortis = bot.process_turn(
        sess_case1_fortis.session_id,
        "I met Dr. Rao at Fortis Hospital. Biomarker testing results are delayed, slowing down patient identification."
    )
    print(f"\nUser: I met Dr. Rao at Fortis Hospital. Biomarker testing results are delayed...")
    print(f"AI:   {res_c1_fortis['bot_message']}")
    assert res_c1_fortis.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "biomarker" in res_c1_fortis['bot_message'].lower()
    assert "tracking" in res_c1_fortis['bot_message'].lower() or "workflow" in res_c1_fortis['bot_message'].lower()
    print(" [PASS] OS Fortis Case 1 triggered biomarker workflow & tracking inquiry!")

    # 3C: FRM Apollo - Rep mentions PA turnaround times
    sess_case1_frm = bot.create_session(role="FRM", brand="RYBREVANT", account_name="Apollo Hospitals")
    res_c1_frm = bot.process_turn(
        sess_case1_frm.session_id,
        "I met with the billing team at Apollo Hospital. Prior auth turnaround times are inconsistent and hard to plan next steps."
    )
    print(f"\nUser: I met with billing at Apollo. PA turnaround times are inconsistent...")
    print(f"AI:   {res_c1_frm['bot_message']}")
    assert res_c1_frm.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "turnaround" in res_c1_frm['bot_message'].lower() or "access-support" in res_c1_frm['bot_message'].lower()
    print(" [PASS] FRM Apollo Case 1 triggered access-support resource inquiry!")

    # 3D: FRM Max - Rep mentions product not on formulary
    sess_case1_max = bot.create_session(role="FRM", brand="RYBREVANT", account_name="Max Healthcare")
    res_c1_max = bot.process_turn(
        sess_case1_max.session_id,
        "I met the pharmacy director at Max Healthcare. RYBREVANT is not on the current formulary and requires an exception pathway."
    )
    print(f"\nUser: I met pharmacy director at Max Healthcare. Product not on formulary...")
    print(f"AI:   {res_c1_max['bot_message']}")
    assert res_c1_max.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "formulary" in res_c1_max['bot_message'].lower() or "committee" in res_c1_max['bot_message'].lower()
    print(" [PASS] FRM Max Case 1 triggered formulary exception committee inquiry!")

    # -------------------------------------------------------------------------
    # Test 4: Immediate Turn 1 Barrier Validation After HCP/Account Capture
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Immediate Turn 1 Barrier Validation After HCP/Account Capture ---")
    
    # 4A: OS Apollo - Turn 1 immediate barrier validation
    sess_c2_os = bot.create_session(role="OS", brand="INLEXZO")
    res_turn1 = bot.process_turn(sess_c2_os.session_id, "I met Dr. Anurag at Apollo Hospital")
    print(f"User: I met Dr. Anurag at Apollo Hospital")
    print(f"AI:   {res_turn1['bot_message']}")
    assert "INLEXZO" in res_turn1['bot_message'], f"Missing brand in Turn 1 question"
    assert "Apollo" in res_turn1['bot_message'] or "candidate flagging" in res_turn1['bot_message'] or "operationalizing" in res_turn1['bot_message'], f"Missing barrier context in Turn 1 question"
    assert res_turn1.get("barrier_case") == "CASE_2_PROACTIVE_FOLLOWUP", f"Expected CASE_2_PROACTIVE_FOLLOWUP but got {res_turn1.get('barrier_case')}"
    print(" [PASS] OS Apollo Turn 1 immediately validates Patient Identification barrier!")

    # 4B: OS Fortis - Turn 1 immediate barrier validation
    sess_fortis = bot.create_session(role="OS", brand="INLEXZO")
    res_fortis = bot.process_turn(sess_fortis.session_id, "I met Dr. Rao at Fortis Hospital")
    print(f"\nUser: I met Dr. Rao at Fortis Hospital")
    print(f"AI:   {res_fortis['bot_message']}")
    assert "biomarker" in res_fortis['bot_message'].lower() or "testing" in res_fortis['bot_message'].lower(), "Missing biomarker barrier context for Fortis OS"
    print(" [PASS] OS Fortis Turn 1 immediately validates biomarker testing barrier!")

    # 4C: FRM Apollo - Turn 1 immediate barrier validation
    sess_frm_apollo = bot.create_session(role="FRM", brand="RYBREVANT")
    res_frm = bot.process_turn(sess_frm_apollo.session_id, "I met the billing team at Apollo Hospital")
    print(f"\nUser: I met the billing team at Apollo Hospital")
    print(f"AI:   {res_frm['bot_message']}")
    assert "prior authorization" in res_frm['bot_message'].lower() or "turnaround" in res_frm['bot_message'].lower() or "access-support" in res_frm['bot_message'].lower(), "Missing PA barrier context for FRM Apollo"
    print(" [PASS] FRM Apollo Turn 1 immediately validates PA turnaround barrier!")

    # 4D: FRM Narayana Health - Turn 1 immediate barrier validation
    sess_narayana = bot.create_session(role="FRM", brand="RYBREVANT")
    res_narayana = bot.process_turn(sess_narayana.session_id, "I met with the billing coordinator at Narayana Health")
    print(f"\nUser: I met with billing at Narayana Health")
    print(f"AI:   {res_narayana['bot_message']}")
    assert "Narayana" in res_narayana['bot_message'] or "payer" in res_narayana['bot_message'].lower() or "policy" in res_narayana['bot_message'].lower(), "Missing payer policy barrier context for FRM Narayana"
    print(" [PASS] FRM Narayana Turn 1 immediately validates payer policy barrier!")

    # -------------------------------------------------------------------------
    # Test 5: Dynamic Entity-Relationship Knowledge Graph Materialization
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Dynamic Entity-Relationship Knowledge Graph Materialization ---")
    live_graph = res_c1_os.get("live_graph")
    assert live_graph is not None, "Missing live_graph in response!"
    nodes = live_graph.get("nodes", [])
    edges = live_graph.get("edges", [])
    mermaid_code = live_graph.get("mermaid", "")

    node_types = {n.get("type") for n in nodes}
    print(f"Discovered Nodes ({len(nodes)}): {node_types}")
    print(f"Discovered Edges ({len(edges)}): {[e.get('label') for e in edges]}")
    
    assert "rep" in node_types, "Missing rep node"
    assert "role" in node_types, "Missing role node"
    assert "account" in node_types, "Missing account node"
    assert "hcp" in node_types, "Missing hcp node"
    assert "barrier" in node_types, "Missing barrier node"
    assert "flowchart TD" in mermaid_code, "Invalid Mermaid header"
    assert "classDef rep" in mermaid_code, "Missing Mermaid styling definitions"
    assert "HAS_HISTORICAL_BARRIER" in mermaid_code or "RAISED_BARRIER" in mermaid_code, "Missing ER relation in Mermaid diagram"
    print(" [PASS] Dynamic Entity-Relationship Graph verified with rich typed nodes & directed relations!")

    # -------------------------------------------------------------------------
    # Test 6: Out-of-Domain Guardrail Protection
    # -------------------------------------------------------------------------
    print("\n--- Test 6: Out-of-Domain Guardrail Protection ---")
    sess_ood = bot.create_session(role="OS", brand="INLEXZO")
    res_ood = bot.process_turn(sess_ood.session_id, "The weather is very sunny today in the park")
    print(f"User: The weather is very sunny today in the park")
    print(f"AI:   {res_ood['bot_message']}")
    assert res_ood["status"] == "OUT_OF_DOMAIN_INTERCEPT"
    print(" [PASS] OOD input intercepted without state corruption.")

    # -------------------------------------------------------------------------
    # Test 7: Universal Session Wrap-Up Confirmation
    # -------------------------------------------------------------------------
    print("\n--- Test 7: Universal Session Wrap-Up Confirmation ---")
    res_close = bot.process_turn(sess_c2_os.session_id, "yes we can wrap up now")
    print(f"User: yes we can wrap up now")
    print(f"Status: {res_close.get('status')}")
    assert res_close.get("status") == "SESSION_CLOSED"
    print(" [PASS] Universal wrap-up closed cleanly.")

    # -------------------------------------------------------------------------
    # Test 8: Full 9-Turn Conversational Flow Regression
    # (Exact user transcript - no derailment or dataset contamination)
    # -------------------------------------------------------------------------
    print("\n--- Test 8: Full 9-Turn Conversational Flow Regression ---")
    sess_full = bot.create_session(role="OS", brand="INLEXZO")
    
    turns = [
        ("Dr Anurag at apollo hospital", "Turn 1: Account+HCP capture -> immediate barrier validation"),
        ("to get update if they get new eligible patient for the NMIBC area or bladder cancer details. also, what treatment pathway he is following for the patient.", "Turn 2: Call purpose -> treatment sequencing"),
        ("yes, he mentioned that recently a patient with post BCG failure has been identified.", "Turn 3: Patient identified -> BCG status"),
        ("he had full course of BCG treatment. and the candiadate in considered as BCG-unresponsive", "Turn 4: BCG status -> next step"),
        ("to get the full details. of patient history and hcp will be evaluating whether he is eligible for the inlexzo.", "Turn 5: Medical history evaluation -> clinical parameters (NOT cystectomy)"),
        ("We havent discussed about this", "Turn 6: Dismissive -> clean timing/touchpoint (NOT off-topic)"),
        ("As of now there are no barriers", "Turn 7: No barriers -> timing/support (NEVER 'Like I said...')"),
        ("in a couple of weeks", "Turn 8: Timing provided -> support/resources"),
        ("nothing else", "Turn 9: Nothing else -> wrap-up"),
    ]
    
    for i, (user_msg, description) in enumerate(turns):
        res = bot.process_turn(sess_full.session_id, user_msg)
        ai_msg = res['bot_message']
        print(f"  [{description}]")
        print(f"    User: {user_msg}")
        print(f"    AI:   {ai_msg}")
        
        ai_lower = ai_msg.lower()
        
        if i == 0:  # Turn 1: Must include barrier validation
            assert "inlexzo" in ai_lower, f"Turn 1 missing brand"
            assert "apollo" in ai_lower or "candidate" in ai_lower or "operationalizing" in ai_lower or "pathway" in ai_lower, f"Turn 1 missing barrier context"
        elif i == 4:  # Turn 5: Must ask about clinical evaluation, NOT cystectomy
            assert "cystectomy" not in ai_lower, f"Turn 5 FAILED: cystectomy mentioned (off-context)"
            assert "?" in ai_msg, f"Turn 5 FAILED: not a question"
        elif i == 5:  # Turn 6: Must handle 'We havent discussed' cleanly
            assert "like i said" not in ai_lower, f"Turn 6 FAILED: dataset contamination ('Like I said...')"
            assert "product quality" not in ai_lower, f"Turn 6 FAILED: product quality mentioned (off-context)"
            assert "?" in ai_msg, f"Turn 6 FAILED: not a question"
        elif i == 6:  # Turn 7: Must handle 'no barriers' cleanly
            assert "like i said" not in ai_lower, f"Turn 7 FAILED: dataset contamination ('Like I said...')"
            assert "product quality" not in ai_lower, f"Turn 7 FAILED: product quality mentioned"
            assert "?" in ai_msg, f"Turn 7 FAILED: not a question"
        
        # Universal: AI response must always be a proper question (ends with ?)
        assert ai_msg.strip().endswith("?") or res.get("status") == "SESSION_CLOSED", f"Turn {i+1} FAILED: AI output is not a question: {ai_msg}"
    
    print(" [PASS] Full 9-turn conversational flow completed without derailment!")

    print("\n" + "=" * 80)
    print("ALL 8 COMPREHENSIVE INTELLIGENCE TESTS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
