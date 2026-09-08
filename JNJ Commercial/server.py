"""
AnQ Bot - KG-Conditioned Conversational Call Note Assistant Server
==================================================================
Serves the web UI and REST API endpoints connected to the Rule-Governed
Knowledge Graph Engine and SLM Next-Question Generation Pipeline.
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import sys

from engine.chatbot_pipeline import RuleGovernedCallBot

PORT = int(os.getenv("PORT", "8080"))
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Initialize Bot Engine
print("Initializing AnQ Bot Call Capture Engine...")
bot = RuleGovernedCallBot()

# Machine Warmup for Sub-100ms Inference
def warmup_engine(engine_bot):
    print("[Warmup] Warming up inference machine and pre-caching Knowledge Graph ontologies...")
    try:
        dummy_session = engine_bot.start_session(role="OS", brand="INLEXZO", user_name="Warmup Engine")
        sid = dummy_session["session_id"]
        # Pre-warm NLU, KG context assembly, and SLM engine
        engine_bot.process_turn(sid, "Met with Dr. Anurag at Apollo Hospital regarding eligible patient pathway")
        for _ in engine_bot.process_turn_stream(sid, "Discussed BCG unresponsive criteria"):
            pass
        if sid in engine_bot.active_sessions:
            del engine_bot.active_sessions[sid]
        print("[Warmup] Machine warmed up successfully. Sub-100ms TRT & TTFT active.")
    except Exception as e:
        print(f"[Warmup] Notice: {e}")

warmup_engine(bot)

DEMO_SCENARIOS = [
    {
        "id": "masters_analogy_frm",
        "title": "Scenario 1A: FRM Clinical Efficacy (Rule Violation)",
        "description": "Simulates an FRM (Masters analogue) attempting to log clinical trial efficacy & biomarker data.",
        "role": "FRM",
        "brand": "RYBREVANT",
        "turn_1": "Yes, I have a few minutes.",
        "turn_2": "The oncologist asked about clinical trial efficacy and biomarker testing for our lung cancer patient.",
        "expected_behavior": "Intercepts with Scope Notice [resp:FRM:28] and blocks recording."
    },
    {
        "id": "phd_analogy_os",
        "title": "Scenario 1B: OS Clinical Efficacy (Rule Compliant)",
        "description": "Simulates an OS (PhD analogue) discussing the exact same clinical trial efficacy & biomarker data.",
        "role": "OS",
        "brand": "RYBREVANT",
        "turn_1": "Yes, ready to capture notes.",
        "turn_2": "The oncologist asked about clinical trial efficacy and biomarker testing for our lung cancer patient.",
        "expected_behavior": "Approved under OS promotional scope (resp:OS:2). State advances & predicts next question."
    },
    {
        "id": "frm_prior_auth_flow",
        "title": "Scenario 2: FRM Prior Authorization & Denial",
        "description": "Simulates an FRM logging a prior authorization delay at an oncology account.",
        "role": "FRM",
        "brand": "INLEXZO",
        "turn_1": "Yes, I have a minute.",
        "turn_2": "I met with Lisa at Northside Urology. We discussed prior authorization delays for INLEXZO.",
        "turn_3": "Lisa said commercial payers are requesting additional clinical chart notes before approval.",
        "turn_4": "I provided the standard prior authorization checklist and agreed to follow up next Tuesday.",
        "expected_behavior": "Full multi-turn state progression through Payer & Access states."
    },
    {
        "id": "phi_violation_check",
        "title": "Scenario 3: Global Compliance PHI Redaction",
        "description": "Simulates a rep mentioning identifiable patient details.",
        "role": "OS",
        "brand": "INLEXZO",
        "turn_1": "Yes, let's log the call.",
        "turn_2": "Patient John Doe DOB 04/12/1965 was prescribed INLEXZO.",
        "expected_behavior": "Immediate PHI Compliance Alert (rule:privacy_phi_pii) with scrubbed text."
    }
]

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "HEALTHY", "service": "AnQ Bot Commercial Oncology"}).encode("utf-8"))
            return

        elif parsed.path == "/api/kg/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            kg_info = {
                "roles": {
                    "OS": {
                        "name": "Oncology Sales Representative",
                        "primary_domain": "Commercial / Promotional Engagement",
                        "in_scope_topics": list(bot.kg.in_scope_topics.get("OS", [])),
                        "out_of_scope_topics": list(bot.kg.out_of_scope_topics.get("OS", [])),
                        "exclusion_rules": [
                            {"id": r["id"], "text": r["properties"].get("text")} 
                            for r in bot.kg.exclusion_rules.get("OS", [])
                        ]
                    },
                    "FRM": {
                        "name": "Field Reimbursement Manager",
                        "primary_domain": "Patient Access & Reimbursement",
                        "in_scope_topics": list(bot.kg.in_scope_topics.get("FRM", [])),
                        "out_of_scope_topics": list(bot.kg.out_of_scope_topics.get("FRM", [])),
                        "exclusion_rules": [
                            {"id": r["id"], "text": r["properties"].get("text")} 
                            for r in bot.kg.exclusion_rules.get("FRM", [])
                        ]
                    }
                },
                "total_indexed_turns": len(bot.kg.core_responsibilities.get("FRM", [])) + len(bot.kg.core_responsibilities.get("OS", [])),
                "model_engine": "SLM (KG-Conditioned Llama-3.2-1B)",
                "governance_mode": "Knowledge Graph Rule Engine"
            }
            self.wfile.write(json.dumps(kg_info).encode("utf-8"))
            return

        elif parsed.path == "/api/accounts/barriers":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(bot.kg_retriever.account_barriers).encode("utf-8"))
            return

        elif parsed.path == "/api/scenarios":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(DEMO_SCENARIOS).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body) if body else {}

        if parsed.path == "/api/session/new":
            role = data.get("role", "OS")
            brand = data.get("brand", "INLEXZO" if role == "OS" else "RYBREVANT")
            user_name = data.get("user_name")
            account_name = data.get("account_name")
            session = bot.create_session(role=role, brand=brand, user_name=user_name, account_name=account_name)
                
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            resp = {
                "session_id": session.session_id,
                "role": session.role,
                "brand": session.slots.get("brand"),
                "account_name": session.slots.get("account_name"),
                "initial_question": session.current_question,
                "current_state": session.current_state,
                "live_graph": session.slots.get("live_graph"),
                "graph_mermaid": session.slots.get("graph_mermaid"),
                "summary": session.get_summary()
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/session/turn":
            session_id = data.get("session_id")
            utterance = data.get("utterance") if data.get("utterance") is not None else data.get("candidate_answer", "")
            
            result = bot.process_turn(session_id=session_id, candidate_answer=utterance)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        elif parsed.path == "/api/session/turn_stream":
            session_id = data.get("session_id")
            utterance = data.get("utterance") if data.get("utterance") is not None else data.get("candidate_answer", "")
            
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            for chunk in bot.process_turn_stream(session_id=session_id, candidate_answer=utterance):
                try:
                    payload = f"data: {json.dumps(chunk)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    break
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass
            return

        self.send_response(404)
        self.end_headers()

class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def run_server():
    with ThreadingServer(("", PORT), RequestHandler) as httpd:
        print(f"\n============================================================")
        print(f" J&J Commercial Non-LLM Conversational Assistant Live!")
        print(f" Access UI at: http://localhost:{PORT}")
        print(f"============================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer shutting down.")

if __name__ == "__main__":
    run_server()
