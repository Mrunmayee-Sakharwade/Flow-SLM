"""
test_embedding_kg_pipeline.py
==============================
Dedicated Test Suite for Dynamic Knowledge Graph Embedding Integration (OS Scoped):
1. Dynamic KG Context Keeping & Intent Retrieval for Novel/Arbitrary Utterances
2. Telemetry Verification: TTFT (ms), TRT (ms), generated_tokens, num_tokens, tokens_per_second
3. Case 1 & Case 2 Account Friction Context Integration with Dynamic KG Matching
4. Persona Context (OS / INLEXZO) & Regulatory Scope Enforcement
5. Streaming Turn (process_turn_stream) Metrics Verification
"""

import time
from engine.chatbot_pipeline import RuleGovernedCallBot
from engine.kg_embedding_engine import KGEmbeddingEngine
from engine.kg_context_retriever import KGContextRetriever

def test_dynamic_embedding_kg():
    print("=" * 80)
    print("DYNAMIC EMBEDDING KNOWLEDGE GRAPH & TELEMETRY VALIDATION SUITE (OS)")
    print("=" * 80)

    bot = RuleGovernedCallBot()

    # -------------------------------------------------------------------------
    # Test 1: Telemetry Verification on Standard Turn
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Telemetry Verification (TTFT, TRT, Tokens) ---")
    sess1 = bot.create_session(role="OS", brand="INLEXZO", account_name="Atlantic Urology Associates")
    res1 = bot.process_turn(
        sess1.session_id,
        "We reviewed the patient selection criteria for BCG-unresponsive NMIBC with Dr. Robert Patel today."
    )

    print(f"User utterance: 'We reviewed the patient selection criteria for BCG-unresponsive NMIBC...'")
    print(f"Predicted Question: {res1['bot_message']}")
    print(f"Telemetry metrics:")
    print(f"  - ttft_ms          : {res1.get('ttft_ms')} ms")
    print(f"  - trt_ms           : {res1.get('trt_ms')} ms")
    print(f"  - generated_tokens : {res1.get('generated_tokens')}")
    print(f"  - num_tokens       : {res1.get('num_tokens')}")
    print(f"  - tokens_per_second: {res1.get('tokens_per_second')}")

    # Check root level metrics
    assert "ttft_ms" in res1 and res1["ttft_ms"] is not None, "Missing root ttft_ms"
    assert "trt_ms" in res1 and res1["trt_ms"] is not None, "Missing root trt_ms"
    assert "generated_tokens" in res1 and res1["generated_tokens"] > 0, "Missing root generated_tokens"
    assert "num_tokens" in res1 and res1["num_tokens"] > 0, "Missing root num_tokens"

    # Check nested metrics dictionary
    m = res1.get("metrics", {})
    assert "ttft_ms" in m, "Missing metrics.ttft_ms"
    assert "trt_ms" in m, "Missing metrics.trt_ms"
    assert "generated_tokens" in m, "Missing metrics.generated_tokens"
    print(" [PASS] Telemetry (TTFT, TRT, Tokens) verified at root and metrics object!")

    # -------------------------------------------------------------------------
    # Test 2: Dynamic Semantic Intent Retrieval on Unscripted Phrasings
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Dynamic Semantic Intent Retrieval on Unscripted Input ---")
    unscripted_cases = [
        ("OS", "INLEXZO", "Atlantic Urology Associates", 
         "I met with Dr. Robert Patel at Atlantic Urology. The clinic is rechecking coverage policy guidelines before setting insertion dates.",
         ["coverage", "benefits", "rechecking", "patient", "scheduling", "inlexzo", "patel"]),
        ("OS", "INLEXZO", "Northside Urology Group",
         "I met Dr. Sarah Jenkins at Northside Urology. The PA submission is currently in process with the insurer.",
         ["prior authorization", "pa", "approval", "clears", "status", "jenkins", "inlexzo"]),
        ("OS", "INLEXZO", "Capital Bladder Cancer Center",
         "I met Dr. Marcus Vance at Capital Bladder Cancer Center. The hospital P&T committee meets next month to evaluate formulary placement.",
         ["p&t", "committee", "timing", "deductible", "vance", "inlexzo", "formulary"])
    ]

    for role, brand, acc, user_text, expected_keywords in unscripted_cases:
        sess = bot.create_session(role=role, brand=brand, account_name=acc)
        res = bot.process_turn(sess.session_id, user_text)
        bot_msg = res["bot_message"]
        print(f"\n[{role}] User: '{user_text}'")
        print(f"AI Predicted Question: '{bot_msg}'")
        assert any(k.lower() in bot_msg.lower() for k in expected_keywords), (
            f"Expected at least one of {expected_keywords} in bot response: '{bot_msg}'"
        )
        assert res.get("ttft_ms") is not None
        assert res.get("trt_ms") is not None
        print(f" [PASS] Unscripted {role} intent dynamically matched and steered question!")

    # -------------------------------------------------------------------------
    # Test 3: Streaming Telemetry Verification (process_turn_stream)
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Streaming Telemetry Verification ---")
    sess_stream = bot.create_session(role="OS", brand="INLEXZO", account_name="Valley Urology Specialists")
    stream_generator = bot.process_turn_stream(
        sess_stream.session_id,
        "I met Dr. Lisa Wong at Valley Urology Specialists. We reviewed anatomical model prep and procedure setup."
    )

    tokens = []
    stream_metrics = None
    stream_complete = None
    for chunk in stream_generator:
        if chunk.get("type") == "token":
            tokens.append(chunk.get("token", ""))
        elif chunk.get("type") == "metrics":
            stream_metrics = chunk
        elif chunk.get("type") in ["result", "complete"]:
            stream_complete = chunk

    streamed_text = "".join(tokens).strip()
    print(f"Streamed Question: {streamed_text}")
    print(f"Stream metrics chunk: {stream_metrics}")
    print(f"Stream complete chunk: {stream_complete.get('latency_ms')} ms")

    assert stream_metrics is not None, "Missing stream metrics event"
    assert "ttft_ms" in stream_metrics, "Missing stream ttft_ms"
    assert "trt_ms" in stream_metrics, "Missing stream trt_ms"
    assert "generated_tokens" in stream_metrics, "Missing stream generated_tokens"
    assert stream_complete is not None, "Missing stream complete event"
    assert stream_complete.get("trt_ms") is not None, "Missing complete trt_ms"
    assert stream_complete.get("ttft_ms") is not None, "Missing complete ttft_ms"
    print(" [PASS] Streaming telemetry (TTFT, TRT, token counts) successfully emitted!")

    # -------------------------------------------------------------------------
    # Test 4: Dynamic Account Barrier Matching via MedEmbed
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Dynamic Account Barrier Semantic Matching ---")
    retriever = KGContextRetriever()
    assert retriever.embedder is not None, "KGEmbeddingEngine not attached to retriever"

    # Test arbitrary barrier description matching
    match_atl = retriever.embedder.match_account_barrier(
        "Recent coverage changes caused billing uncertainty and we need benefits verification before scheduling",
        account_name="Atlantic Urology Associates",
        role="OS"
    )
    assert match_atl is not None, "Failed to match Atlantic Urology barrier via embeddings"
    print(f" [PASS] Atlantic Urology Barrier matched with semantic score: {match_atl.get('semantic_score')}")

    match_northside = retriever.embedder.match_account_barrier(
        "Prior authorization submissions are currently in process with payers and calendar booking is blocked",
        account_name="Northside Urology Group",
        role="OS"
    )
    assert match_northside is not None, "Failed to match Northside Urology barrier via embeddings"
    print(f" [PASS] Northside Urology Barrier matched with semantic score: {match_northside.get('semantic_score')}")

    print("\n" + "=" * 80)
    print("ALL DYNAMIC EMBEDDING KG & TELEMETRY TESTS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_dynamic_embedding_kg()
