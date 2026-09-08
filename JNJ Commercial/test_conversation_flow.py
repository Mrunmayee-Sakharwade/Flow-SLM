import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.chatbot_pipeline import RuleGovernedCallBot

def test_user_flow():
    print("Initializing RuleGovernedCallBot...")
    bot = RuleGovernedCallBot()
    
    print("\n--- TEST 1: Exact User Transcript with Hang Up ---")
    session = bot.create_session(role="OS", brand="INLEXZO", user_name="Mihir", account_name="Hospital")
    print(f"Initial AI Greeting: {session.current_question}")
    
    # Turn 1
    res1 = bot.process_turn(session.session_id, "Yes, we can.")
    print(f"\nUser: Yes, we can.")
    print(f"AI: {res1.get('bot_message')} (status: {res1.get('status')})")
    assert "who" in res1.get('bot_message', '').lower()
    
    # Turn 2: Dr. Unrug (should normalize to Dr. Anurag)
    res2 = bot.process_turn(session.session_id, "Dr. Unrug at Hospital.")
    print(f"\nUser: Dr. Unrug at Hospital.")
    print(f"AI: {res2.get('bot_message')} (status: {res2.get('status')})")
    print(f"Slots: {session.slots}")
    assert "Anurag" in str(session.slots.get("hcp_name")) or "Anurag" in res2.get('bot_message', '')
    
    # Turn 3: "Nothing." (should NOT trigger compliance toxicity)
    res3 = bot.process_turn(session.session_id, "Nothing.")
    print(f"\nUser: Nothing.")
    print(f"AI: {res3.get('bot_message')} (status: {res3.get('status')})")
    assert res3.get("status") != "SCOPE_VIOLATION_INTERCEPT"
    assert "toxicity" not in res3.get("bot_message", "").lower()
    assert "side-effect" not in res3.get("bot_message", "").lower()
    
    # Turn 4: "We can hang up the call." (should immediately close session)
    res4 = bot.process_turn(session.session_id, "We can hang up the call.")
    print(f"\nUser: We can hang up the call.")
    print(f"AI: {res4.get('bot_message')} (status: {res4.get('status')}, completed: {res4.get('is_completed')})")
    assert res4.get("status") == "SESSION_CLOSED"
    assert res4.get("is_completed") is True
    assert "who did you meet" not in res4.get("bot_message", "").lower()
    print("✓ TEST 1 PASSED!")

    print("\n--- TEST 2: Saying 'No.' does NOT trigger compliance gate ---")
    session2 = bot.create_session(role="OS", brand="INLEXZO", user_name="Mihir")
    bot.process_turn(session2.session_id, "Yes.")
    bot.process_turn(session2.session_id, "Dr. Anurag at Apollo.")
    res_neg = bot.process_turn(session2.session_id, "No.")
    print(f"\nUser: No.")
    print(f"AI: {res_neg.get('bot_message')} (status: {res_neg.get('status')})")
    assert res_neg.get("status") != "SCOPE_VIOLATION_INTERCEPT"
    assert "toxicity" not in res_neg.get("bot_message", "").lower()
    assert "side-effect" not in res_neg.get("bot_message", "").lower()
    print("✓ TEST 2 PASSED!")

    print("\n--- TEST 3: Wrap-up confirmation with 'Yes!' with punctuation ---")
    session3 = bot.create_session(role="OS", brand="INLEXZO", user_name="Mihir")
    session3.current_state = "STATE_7_WRAP_UP_CONFIRMATION"
    session3.current_question = "Got it. I have the main points. Are we ready to finish?"
    res_wrap = bot.process_turn(session3.session_id, "Yes!")
    print(f"\nUser: Yes!")
    print(f"AI: {res_wrap.get('bot_message')} (status: {res_wrap.get('status')}, completed: {res_wrap.get('is_completed')})")
    assert res_wrap.get("status") == "SESSION_CLOSED"
    assert res_wrap.get("is_completed") is True
    assert "target timing" not in res_wrap.get("bot_message", "").lower()
    print("✓ TEST 3 PASSED!")

    print("\n--- TEST 4: Streaming Turn Session Close ---")
    session4 = bot.create_session(role="OS", brand="INLEXZO", user_name="Mihir")
    bot.process_turn(session4.session_id, "Yes.")
    bot.process_turn(session4.session_id, "Dr. Anurag at Apollo.")
    stream_chunks = list(bot.process_turn_stream(session4.session_id, "We can stop the call."))
    last_chunk = [c for c in stream_chunks if c.get("type") == "result"][0]
    print(f"Stream result status: {last_chunk.get('status')}, completed: {last_chunk.get('is_completed')}")
    assert last_chunk.get("status") == "SESSION_CLOSED"
    assert last_chunk.get("is_completed") is True
    print("✓ TEST 4 PASSED!")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_user_flow()
