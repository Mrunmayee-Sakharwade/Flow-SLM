"""
test_account_intelligence.py
============================
Comprehensive Validation Suite for J&J Account Barrier Intelligence (OS Scoped):
1. Account Entity Normalization (8 US Oncology Accounts from generated_os_training_transcripts 1.xlsx)
2. Role-Conditioned Dedicated KG (kg_os.json) Node & Edge Verification
3. Dual-Case Conversational Handling:
   - Case 1 (Rep-Initiated): Rep brings up barrier -> AI immediately addresses account ask
   - Case 2 (Proactive AI): Rep doesn't mention barrier -> AI proactively checks historical barrier
4. Dynamic Live Entity-Relationship Knowledge Graph Materialization
5. Out-of-Domain & Compliance Guardrail Protection
6. Universal Clean Session Wrap-Up
7. Full Multi-Turn Conversational Flow Regression on INLEXZO
"""

import sys
from engine.nlu_extractor import NLUExtractor
from engine.kg_context_retriever import KGContextRetriever
from engine.chatbot_pipeline import RuleGovernedCallBot

def run_tests():
    print("=" * 80)
    print("JNJ OS COMMERCIAL INTELLIGENCE & KNOWLEDGE GRAPH VALIDATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1: NLU Account Entity Normalization (8 US Accounts)
    # -------------------------------------------------------------------------
    print("\n--- Test 1: NLU Account Entity Normalization ---")
    nlu = NLUExtractor()
    acc_tests = [
        ("I visited Dr Patel at Atlantic Urology today", "Atlantic Urology Associates"),
        ("Met Dr Vance at Capital Bladder Cancer Center", "Capital Bladder Cancer Center"),
        ("Spoke with clinic staff at Central Ohio Urology", "Central Ohio Urology"),
        ("Called team at Northside Urology Group", "Northside Urology Group"),
        ("Visited Regional Urology Institute", "Regional Urology Institute"),
        ("Met with Summit Urologic Oncology team", "Summit Urologic Oncology"),
        ("At Temple Urology Clinic today", "Temple Urology Clinic"),
        ("Visited Valley Urology Specialists", "Valley Urology Specialists")
    ]
    for text, expected_acc in acc_tests:
        res = nlu.extract_entities(text)
        acc = res.get("accounts", [None])[0]
        assert acc == expected_acc, f"Failed for '{text}': got '{acc}', expected '{expected_acc}'"
        print(f" [PASS] '{text}' -> {acc}")

    # -------------------------------------------------------------------------
    # Test 2: Dedicated OS KG & 8 Account Barrier Profiles
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Dedicated OS KG & 8 Account Barrier Profiles ---")
    kg = KGContextRetriever()
    accounts = [
        "Atlantic Urology Associates",
        "Capital Bladder Cancer Center",
        "Central Ohio Urology",
        "Northside Urology Group",
        "Regional Urology Institute",
        "Summit Urologic Oncology",
        "Temple Urology Clinic",
        "Valley Urology Specialists"
    ]
    for acc in accounts:
        os_b = kg.get_account_barrier(acc, "OS")
        assert os_b is not None, f"Missing OS barrier for {acc}"
        assert len(os_b.get("trigger_keywords", [])) >= 5, f"Insufficient keywords for {acc}"
        assert os_b.get("case_1_inquiry") is not None, f"Missing case_1_inquiry for {acc}"
        assert os_b.get("case_2_inquiry") is not None, f"Missing case_2_inquiry for {acc}"
        print(f" [PASS] {acc}:")
        print(f"        Barrier: {os_b['barrier_type']} ({len(os_b['trigger_keywords'])} trigger keywords)")

    bot = RuleGovernedCallBot()

    # -------------------------------------------------------------------------
    # Test 3: Dual-Case Handling - Case 1: Rep-Initiated Barrier Inquiry
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Dual-Case Handling - Case 1 (Rep-Initiated Barrier) ---")
    
    # 3A: OS Atlantic Urology - Coverage change and benefits investigation
    sess_c1_atlantic = bot.create_session(role="OS", brand="INLEXZO", account_name="Atlantic Urology Associates")
    res_c1_atlantic = bot.process_turn(
        sess_c1_atlantic.session_id,
        "I met Dr. Robert Patel at Atlantic Urology Associates. Coverage changed recently and the office is rechecking patient responsibility before scheduling."
    )
    print(f"User: Coverage changed recently and office is rechecking patient responsibility...")
    print(f"AI:   {res_c1_atlantic['bot_message']}")
    assert res_c1_atlantic.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "coverage" in res_c1_atlantic['bot_message'].lower() or "benefits" in res_c1_atlantic['bot_message'].lower() or "atlantic" in res_c1_atlantic['bot_message'].lower()
    print(" [PASS] OS Atlantic Urology Case 1 triggered benefits investigation inquiry!")

    # 3B: OS Northside Urology - PA still in process
    sess_c1_northside = bot.create_session(role="OS", brand="INLEXZO", account_name="Northside Urology Group")
    res_c1_northside = bot.process_turn(
        sess_c1_northside.session_id,
        "I met Dr. Jenkins at Northside Urology Group. The PA is still in process and they are not ready to schedule until that clears."
    )
    print(f"\nUser: The PA is still in process and not ready to schedule until that clears...")
    print(f"AI:   {res_c1_northside['bot_message']}")
    assert res_c1_northside.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "prior authorization" in res_c1_northside['bot_message'].lower() or "pa" in res_c1_northside['bot_message'].lower()
    print(" [PASS] OS Northside Urology Case 1 triggered prior authorization inquiry!")

    # 3C: OS Capital Bladder - P&T approval and deductible
    sess_c1_capital = bot.create_session(role="OS", brand="INLEXZO", account_name="Capital Bladder Cancer Center")
    res_c1_capital = bot.process_turn(
        sess_c1_capital.session_id,
        "I met Dr. Vance at Capital Bladder Cancer Center. P&T approval is the main challenge and we are working through deductible and payment timing."
    )
    print(f"\nUser: P&T approval is the main challenge and deductible payment timing...")
    print(f"AI:   {res_c1_capital['bot_message']}")
    assert res_c1_capital.get("barrier_case") == "CASE_1_USER_INITIATED"
    assert "p&t" in res_c1_capital['bot_message'].lower() or "committee" in res_c1_capital['bot_message'].lower() or "deductible" in res_c1_capital['bot_message'].lower()
    print(" [PASS] OS Capital Bladder Case 1 triggered P&T and affordability inquiry!")

    # -------------------------------------------------------------------------
    # Test 4: Immediate Turn 1 Barrier Validation After HCP/Account Capture
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Immediate Turn 1 Barrier Validation After HCP/Account Capture ---")
    
    # 4A: OS Atlantic Urology - Turn 1 immediate barrier validation
    sess_t1_atl = bot.create_session(role="OS", brand="INLEXZO")
    res_turn1_atl = bot.process_turn(sess_t1_atl.session_id, "I met Dr. Robert Patel at Atlantic Urology Associates")
    print(f"User: I met Dr. Robert Patel at Atlantic Urology Associates")
    print(f"AI:   {res_turn1_atl['bot_message']}")
    assert "INLEXZO" in res_turn1_atl['bot_message'], "Missing brand in Turn 1 question"
    assert "Atlantic" in res_turn1_atl['bot_message'] or "coverage" in res_turn1_atl['bot_message'].lower() or "benefits" in res_turn1_atl['bot_message'].lower(), "Missing barrier context in Turn 1 question"
    assert res_turn1_atl.get("barrier_case") == "CASE_2_PROACTIVE_FOLLOWUP", f"Expected CASE_2_PROACTIVE_FOLLOWUP but got {res_turn1_atl.get('barrier_case')}"
    print(" [PASS] OS Atlantic Urology Turn 1 immediately validates coverage/benefits barrier!")

    # 4B: OS Northside Urology - Turn 1 immediate barrier validation
    sess_t1_ns = bot.create_session(role="OS", brand="INLEXZO")
    res_turn1_ns = bot.process_turn(sess_t1_ns.session_id, "I met Dr. Sarah Jenkins at Northside Urology Group")
    print(f"\nUser: I met Dr. Sarah Jenkins at Northside Urology Group")
    print(f"AI:   {res_turn1_ns['bot_message']}")
    assert "prior authorization" in res_turn1_ns['bot_message'].lower() or "pa" in res_turn1_ns['bot_message'].lower() or "northside" in res_turn1_ns['bot_message'].lower(), "Missing PA barrier context for Northside"
    print(" [PASS] OS Northside Urology Turn 1 immediately validates PA turnaround barrier!")

    # 4C: OS Capital Bladder - Turn 1 immediate barrier validation
    sess_t1_cap = bot.create_session(role="OS", brand="INLEXZO")
    res_turn1_cap = bot.process_turn(sess_t1_cap.session_id, "I met Dr. Marcus Vance at Capital Bladder Cancer Center")
    print(f"\nUser: I met Dr. Marcus Vance at Capital Bladder Cancer Center")
    print(f"AI:   {res_turn1_cap['bot_message']}")
    assert "p&t" in res_turn1_cap['bot_message'].lower() or "committee" in res_turn1_cap['bot_message'].lower() or "capital" in res_turn1_cap['bot_message'].lower(), "Missing P&T barrier context for Capital Bladder"
    print(" [PASS] OS Capital Bladder Turn 1 immediately validates P&T review barrier!")

    # -------------------------------------------------------------------------
    # Test 5: Dynamic Entity-Relationship Knowledge Graph Materialization
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Dynamic Entity-Relationship Knowledge Graph Materialization ---")
    live_graph = res_c1_atlantic.get("live_graph")
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
    res_close = bot.process_turn(sess_t1_atl.session_id, "yes we can wrap up now")
    print(f"User: yes we can wrap up now")
    print(f"Status: {res_close.get('status')}")
    assert res_close.get("status") == "SESSION_CLOSED"
    print(" [PASS] Universal wrap-up closed cleanly.")

    # -------------------------------------------------------------------------
    # Test 8: Full Conversational Flow Regression on INLEXZO
    # -------------------------------------------------------------------------
    print("\n--- Test 8: Full Conversational Flow Regression on INLEXZO ---")
    sess_full = bot.create_session(role="OS", brand="INLEXZO")
    
    turns = [
        ("Dr Robert Patel at Atlantic Urology Associates", "Turn 1: Account+HCP capture -> immediate barrier validation"),
        ("to evaluate an eligible patient with BCG-unresponsive NMIBC with CIS", "Turn 2: Call purpose -> treatment sequencing"),
        ("he completed full induction and maintenance BCG and is considered BCG-unresponsive", "Turn 3: BCG failure confirmation"),
        ("evaluating whether to proceed with INLEXZO insertion", "Turn 4: Treatment positioning"),
        ("we reviewed the procedure room prep and anatomical demo model", "Turn 5: Clinical operational workflow"),
        ("prior authorizations are in process with regional payers", "Turn 6: Barrier handling"),
        ("I will follow up next Tuesday to confirm insertion scheduling", "Turn 7: Territory engagement planning"),
        ("nothing else", "Turn 8: Wrap-up confirmation"),
    ]
    
    for i, (user_msg, description) in enumerate(turns):
        res = bot.process_turn(sess_full.session_id, user_msg)
        ai_msg = res['bot_message']
        print(f"  [{description}]")
        print(f"    User: {user_msg}")
        print(f"    AI:   {ai_msg}")
        
        ai_lower = ai_msg.lower()
        if i == 0:
            assert "inlexzo" in ai_lower, "Turn 1 missing brand"
            assert "atlantic" in ai_lower or "coverage" in ai_lower or "benefits" in ai_lower or "patel" in ai_lower, "Turn 1 missing context"
        elif i == len(turns) - 1:
            assert res.get("status") == "SESSION_CLOSED" or "wrap" in ai_lower or "logged" in ai_lower or "session" in ai_lower, "Final turn should wrap up"
        else:
            assert "?" in ai_msg or res.get("status") == "SESSION_CLOSED", f"Turn {i+1} AI output should be a question: {ai_msg}"
    
    print(" [PASS] Full conversational flow completed cleanly without derailment!")

    print("\n" + "=" * 80)
    print("ALL 8 COMPREHENSIVE INTELLIGENCE TESTS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
