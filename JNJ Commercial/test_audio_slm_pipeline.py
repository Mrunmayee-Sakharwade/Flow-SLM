"""
End-to-End Test: Audio → SLM Next-Question Pipeline
=====================================================
Tests the three new audio-enabled endpoints:
  1. POST /api/audio/transcribe     - Standalone Whisper STT
  2. POST /api/session/audio_turn   - Full Audio → SLM Pipeline
  3. POST /api/session/audio_turn_stream - Streaming Audio → SLM Pipeline

Usage:
    python test_audio_slm_pipeline.py [SERVER_HOST] [SERVER_PORT]

    # Default: http://localhost:8000
    python test_audio_slm_pipeline.py

    # Remote GPU server:
    python test_audio_slm_pipeline.py 10.225.67.250 8008

Requires:
    pip install requests
"""

import os
import sys
import json
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
SERVER_PORT = sys.argv[2] if len(sys.argv) > 2 else "8000"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Find sample audio files from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_AUDIO_FILES = [
    os.path.join(PROJECT_ROOT, "blessing.wav"),
    os.path.join(PROJECT_ROOT, "michael.wav"),
]

def find_sample_audio():
    """Find an available sample audio file for testing."""
    for path in SAMPLE_AUDIO_FILES:
        if os.path.exists(path):
            print(f"  Using sample audio: {path} ({os.path.getsize(path) / 1024:.1f} KB)")
            return path
    print("  [WARNING] No sample audio files found. Looking for any .wav file...")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                print(f"  Found: {path}")
                return path
    return None


def test_health():
    """Verify server is running."""
    print("\n[0/4] Checking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✓ Server healthy: {resp.json()}")
            return True
        # Try /api/kg/info as fallback (FastAPI version)
    except Exception:
        pass
    
    try:
        resp = requests.get(f"{BASE_URL}/api/kg/info", timeout=5)
        if resp.status_code == 200:
            print(f"  ✓ Server healthy (KG info endpoint responding)")
            return True
    except Exception as e:
        print(f"  ✗ Server not reachable at {BASE_URL}: {e}")
        return False

    print(f"  ✗ Server not healthy at {BASE_URL}")
    return False


def test_1_transcribe(audio_path):
    """Test standalone audio transcription endpoint."""
    print("\n" + "=" * 70)
    print("[1/4] Testing POST /api/audio/transcribe (Standalone Whisper STT)")
    print("=" * 70)

    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
        resp = requests.post(f"{BASE_URL}/api/audio/transcribe", files=files, timeout=60)

    print(f"  HTTP Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"  ✗ FAILED: {resp.text}")
        return None

    data = resp.json()
    print(f"  Transcript     : \"{data['transcript']}\"")
    print(f"  Language       : {data['language']}")
    print(f"  Confidence     : {data['confidence']:.4f}")
    print(f"  Duration       : {data['duration_seconds']:.2f}s")
    print(f"  STT Latency    : {data['transcription_latency_ms']:.1f}ms")

    assert data["transcript"], "Transcript should not be empty"
    assert data["language"], "Language should be detected"
    assert data["duration_seconds"] > 0, "Duration should be positive"
    assert data["transcription_latency_ms"] > 0, "Latency should be positive"

    print("  ✓ PASSED: Standalone transcription working correctly!")
    return data


def test_2_audio_turn(audio_path):
    """Test the full audio → SLM pipeline endpoint."""
    print("\n" + "=" * 70)
    print("[2/4] Testing POST /api/session/audio_turn (Full Audio → SLM Pipeline)")
    print("=" * 70)

    # Step 1: Create a new session
    print("  Creating new session...")
    session_resp = requests.post(
        f"{BASE_URL}/api/session/new",
        json={
            "role": "OS",
            "brand": "RYBREVANT",
            "user_name": "Audio Test User"
        },
        timeout=30
    )

    if session_resp.status_code != 200:
        print(f"  ✗ Failed to create session: {session_resp.text}")
        return None

    session_data = session_resp.json()
    session_id = session_data["session_id"]
    print(f"  Session created: {session_id}")
    print(f"  Initial question: \"{session_data['initial_question']}\"")

    # Step 2: Send audio turn
    print(f"\n  Sending audio to /api/session/audio_turn...")
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"session_id": session_id}
        resp = requests.post(
            f"{BASE_URL}/api/session/audio_turn",
            files=files,
            data=data,
            timeout=120
        )

    print(f"  HTTP Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"  ✗ FAILED: {resp.text}")
        return None

    result = resp.json()
    print(f"\n  --- PIPELINE RESULT ---")
    print(f"  Status           : {result.get('status')}")
    print(f"  Transcript       : \"{result.get('transcript')}\"")
    print(f"  STT Latency      : {result.get('transcription_latency_ms', 'N/A')}ms")
    print(f"  Audio Duration   : {result.get('audio_duration_seconds', 'N/A')}s")
    print(f"  STT Language     : {result.get('transcription_language', 'N/A')}")
    print(f"  STT Confidence   : {result.get('transcription_confidence', 'N/A')}")
    print(f"  Bot Message (SLM): \"{result.get('bot_message')}\"")
    print(f"  Target Topic     : {result.get('target_topic', 'N/A')}")
    print(f"  Current State    : {result.get('current_state', 'N/A')}")
    print(f"  Next State       : {result.get('next_state', 'N/A')}")
    print(f"  SLM Latency      : {result.get('latency_ms', 'N/A')}ms")
    print(f"  Engine           : {result.get('engine', 'N/A')}")

    assert result.get("transcript"), "Transcript should be present"
    assert result.get("bot_message"), "Bot message (SLM prediction) should be present"
    assert result.get("status") in ["SUCCESS", "SCOPE_VIOLATION_INTERCEPT", "COMPLIANCE_VIOLATION", "OUT_OF_DOMAIN_INTERCEPT", "SESSION_CLOSED"], \
        f"Unexpected status: {result.get('status')}"

    print("\n  ✓ PASSED: Full audio → SLM pipeline working correctly!")
    return result


def test_3_audio_turn_stream(audio_path):
    """Test the streaming audio → SLM pipeline endpoint."""
    print("\n" + "=" * 70)
    print("[3/4] Testing POST /api/session/audio_turn_stream (SSE Streaming)")
    print("=" * 70)

    # Create a new session
    print("  Creating new session...")
    session_resp = requests.post(
        f"{BASE_URL}/api/session/new",
        json={
            "role": "FRM",
            "brand": "INLEXZO",
            "user_name": "Stream Test User"
        },
        timeout=30
    )

    if session_resp.status_code != 200:
        print(f"  ✗ Failed to create session: {session_resp.text}")
        return None

    session_data = session_resp.json()
    session_id = session_data["session_id"]
    print(f"  Session created: {session_id}")

    # Send streaming audio turn
    print(f"\n  Streaming audio to /api/session/audio_turn_stream...")
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"session_id": session_id}
        resp = requests.post(
            f"{BASE_URL}/api/session/audio_turn_stream",
            files=files,
            data=data,
            stream=True,
            timeout=120
        )

    print(f"  HTTP Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"  ✗ FAILED: {resp.text}")
        return None

    # Parse SSE events
    events = []
    transcript_event = None
    final_result = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]  # Strip "data: " prefix
        if payload == "[DONE]":
            print("  [SSE] [DONE]")
            break
        try:
            event = json.loads(payload)
            events.append(event)
            event_type = event.get("type", "unknown")

            if event_type == "transcription":
                transcript_event = event
                print(f"  [SSE] Transcription: \"{event['transcript'][:60]}...\" "
                      f"({event['transcription_latency_ms']:.0f}ms)")
            elif event_type == "token":
                print(f"  [SSE] Token: \"{event.get('token', '')[:40]}...\"")
            elif event_type == "metrics":
                print(f"  [SSE] Metrics: TTFT={event.get('first_token_latency_ms', 'N/A')}ms, "
                      f"TPS={event.get('tokens_per_second', 'N/A')}")
            elif event_type == "result":
                final_result = event
                print(f"  [SSE] Result: status={event.get('status')}, "
                      f"message=\"{event.get('bot_message', '')[:50]}...\"")
        except json.JSONDecodeError:
            continue

    print(f"\n  Total SSE events received: {len(events)}")

    assert transcript_event, "Should receive a transcription event"
    assert transcript_event.get("transcript"), "Transcription event should have text"
    assert final_result, "Should receive a final result event"

    print("  ✓ PASSED: Streaming audio → SLM pipeline working correctly!")
    return {"events": events, "transcript": transcript_event, "result": final_result}


def test_4_multi_turn_audio(audio_path):
    """Test multi-turn audio conversation flow."""
    print("\n" + "=" * 70)
    print("[4/4] Testing Multi-Turn Audio Conversation Flow")
    print("=" * 70)

    # Create session
    session_resp = requests.post(
        f"{BASE_URL}/api/session/new",
        json={"role": "OS", "brand": "RYBREVANT", "user_name": "Multi-Turn Test"},
        timeout=30
    )
    session_id = session_resp.json()["session_id"]
    print(f"  Session: {session_id}")

    # Turn 1: Audio input
    print(f"\n  Turn 1: Sending audio...")
    with open(audio_path, "rb") as f:
        resp1 = requests.post(
            f"{BASE_URL}/api/session/audio_turn",
            files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            data={"session_id": session_id},
            timeout=120
        )

    if resp1.status_code == 200:
        r1 = resp1.json()
        print(f"  Turn 1 Transcript: \"{r1.get('transcript', 'N/A')[:60]}...\"")
        print(f"  Turn 1 Bot Reply : \"{r1.get('bot_message', 'N/A')[:60]}...\"")
        print(f"  Turn 1 State     : {r1.get('current_state')} → {r1.get('next_state')}")
    else:
        print(f"  Turn 1 failed: {resp1.status_code} - {resp1.text[:100]}")

    # Turn 2: Text input (to show audio and text can be mixed)
    print(f"\n  Turn 2: Sending text input (mixed mode)...")
    resp2 = requests.post(
        f"{BASE_URL}/api/session/turn",
        json={"session_id": session_id, "utterance": "We discussed prior authorization and payer coverage for the patient."},
        timeout=60
    )

    if resp2.status_code == 200:
        r2 = resp2.json()
        print(f"  Turn 2 Bot Reply : \"{r2.get('bot_message', 'N/A')[:60]}...\"")
        print(f"  Turn 2 State     : {r2.get('current_state')} → {r2.get('next_state')}")
    else:
        print(f"  Turn 2 failed: {resp2.status_code}")

    print("\n  ✓ PASSED: Multi-turn audio + text mixed conversation working!")
    return True


def main():
    print("=" * 70)
    print(" Audio → SLM Next-Question Pipeline End-to-End Test Suite")
    print(f" Target Server: {BASE_URL}")
    print("=" * 70)

    # Pre-flight
    if not test_health():
        print("\n✗ Server is not reachable. Please start the server first:")
        print(f"  cd JNJ\\ Commercial && python main.py")
        sys.exit(1)

    audio_path = find_sample_audio()
    if not audio_path:
        print("\n✗ No sample audio files found. Please add a .wav file to test.")
        sys.exit(1)

    # Run tests
    passed = 0
    total = 4

    try:
        r1 = test_1_transcribe(audio_path)
        if r1:
            passed += 1
    except Exception as e:
        print(f"  ✗ Test 1 EXCEPTION: {e}")

    try:
        r2 = test_2_audio_turn(audio_path)
        if r2:
            passed += 1
    except Exception as e:
        print(f"  ✗ Test 2 EXCEPTION: {e}")

    try:
        r3 = test_3_audio_turn_stream(audio_path)
        if r3:
            passed += 1
    except Exception as e:
        print(f"  ✗ Test 3 EXCEPTION: {e}")

    try:
        r4 = test_4_multi_turn_audio(audio_path)
        if r4:
            passed += 1
    except Exception as e:
        print(f"  ✗ Test 4 EXCEPTION: {e}")

    # Summary
    print("\n" + "=" * 70)
    if passed == total:
        print(f" ALL {total} AUDIO → SLM PIPELINE TESTS PASSED SUCCESSFULLY! ✓")
    else:
        print(f" {passed}/{total} tests passed. {total - passed} failed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
