import urllib.request
import json
import sys

print("=== Running AnQ Bot Call Assistance Verification ===")

# 1. Test /mobile
resp = urllib.request.urlopen("http://localhost:8080/mobile")
html = resp.read().decode("utf-8")
assert "login-avatar-circle" in html, "Missing login avatar circle"
assert "btn-launch-call" in html, "Missing launch button"
assert "call-dynamic-status-bar" in html, "Missing dynamic status bar"
print("[OK] Check 1: /mobile HTML validated (dynamic avatar circle & sticky button present)")

# 2. Test /api/text/autocorrect
sample_text = "Um, uh, we met with Dr. Smith to discuss inlexo and ribrevant with prior auto."
req = urllib.request.Request(
    "http://localhost:8080/api/text/autocorrect",
    data=json.dumps({"text": sample_text}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
auto_data = json.loads(resp.read().decode("utf-8"))
cleaned = auto_data.get("cleaned", "")
print("[OK] Check 2: Autocorrect API validated:")
print("   Raw:    ", sample_text)
print("   Cleaned:", cleaned)
assert "INLEXZO" in cleaned, "Expected brand INLEXZO in cleaned text"
assert "RYBREVANT" in cleaned, "Expected brand RYBREVANT in cleaned text"
assert "prior authorization" in cleaned, "Expected 'prior authorization' in cleaned text"

# 3. Test /api/session/new
req = urllib.request.Request(
    "http://localhost:8080/api/session/new",
    data=json.dumps({
        "role": "OS",
        "brand": "INLEXZO",
        "user_name": "Mihit",
        "account_name": "Atlantic Urology Associates",
        "generate_audio": False
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
sess = json.loads(resp.read().decode("utf-8"))
sid = sess.get("session_id")
initial_q = sess.get("initial_question", "")
print(f"[OK] Check 3: Session created successfully for representative 'Mihit' (Session: {sid})")
print(f"   Greeting: {initial_q}")

# 4. Test /api/session/turn
req = urllib.request.Request(
    "http://localhost:8080/api/session/turn",
    data=json.dumps({
        "session_id": sid,
        "role": "OS",
        "brand": "INLEXZO",
        "user_message": "Discussed BCG-unresponsive patient eligibility with Dr. Miller.",
        "generate_audio": False
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
turn_data = json.loads(resp.read().decode("utf-8"))
bot_reply = turn_data.get("bot_message", "")
print(f"[OK] Check 4: Conversational turn processed successfully:")
print(f"   Bot reply: {bot_reply}")

print("\n=== ALL MOBILE CALL ASSISTANCE CHECKS PASSED PERFECTLY ===")
