"""
End-to-End Simulation Test for Friend's Exact Transcript: Dr Ravi @ Apollo Hyderabad (NMIBC / INLEXZO)
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from engine.chatbot_pipeline import RuleGovernedCallBot

def test_friend_session():
    print("=" * 80)
    print("RUNNING E2E TEST FOR FRIEND'S EXACT MULTI-TURN TRANSCRIPT")
    print("=" * 80)
    
    bot = RuleGovernedCallBot()
    
    # 1. Start OS Session
    session = bot.create_session(role="OS")
    session_id = session.session_id
    print(f"\nAI: {session.current_question}")
    
    # Utterances from friend's transcript
    turns = [
        "yes i am",
        "I ahve met dr ravi at apollo hyderabad",
        "to get an update if they have any eligible patient for NMIBC area, bladder cancer. Also whta kind of treatment pathway they are following",
        "yes, he mentioned that reecntly a patient with post BCG failure has been identified",
        "patient had completed full course of BCG and he is right now conisdered as BCG-unresponsive",
        "No we havent discussed about this",
        "to check the complete medical history top chekck whetehr hge is sutable for inlexzo",
        "no there arent any",
        "they were aiming to decide the treatment approcah in next couple of weeks",
        "he did not mention any thing at the moment",
        "we will saty in touch to get the updated for the patient",
        "this is all we have",
        "yes"
    ]
    
    for idx, user_input in enumerate(turns):
        print(f"\n[Turn {idx+1}] User: {user_input}")
        res = bot.process_turn(session_id=session_id, candidate_answer=user_input)
        
        status = res.get("status")
        bot_msg = res.get("bot_message")
        target_topic = res.get("target_topic", "N/A")
        engine = res.get("engine", "N/A")
        
        print(f"AI: {bot_msg}")
        print(f"    (Status: {status} | Target Topic: {target_topic} | Engine: {engine})")
        
        if res.get("session_summary"):
            slots = res["session_summary"].get("slots", {})
            print(f"    (Slots Extracted -> Brand: {slots.get('brand')}, HCP: {slots.get('hcp_name')}, Account: {slots.get('account_name')})")
            
        if res.get("is_completed") or status == "SESSION_CLOSED":
            print("\n>>> Session successfully closed gracefully!")
            break

    print("\n" + "=" * 80)
    print("FRIEND'S TRANSCRIPT TEST COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    test_friend_session()
