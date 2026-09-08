import urllib.request
import json

def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

# 1. Initialize session
init_data = post_json("http://localhost:8080/api/session/new", {
    "role": "OS",
    "brand": "INLEXZO",
    "user_name": "Aniruddha Joshi"
})
sess_id = init_data["session_id"]
print("--- Step 1: Initial Greeting ---")
print("AI:", init_data["initial_question"])

# 2. Turn 1: Yes we can
t1 = post_json("http://localhost:8080/api/session/turn", {
    "session_id": sess_id,
    "utterance": "Yes we can"
})
print("\n--- Step 2: Rep says 'Yes we can' ---")
print("AI:", t1["bot_message"])

# 3. Turn 2: Dr Anurag at apollo hospital
t2 = post_json("http://localhost:8080/api/session/turn", {
    "session_id": sess_id,
    "utterance": "Dr Anurag at apollo hospital"
})
print("\n--- Step 3: Rep says 'Dr Anurag at apollo hospital' ---")
print("AI:", t2["bot_message"])
live_graph = t2.get("live_graph", {})
nodes = live_graph.get("nodes", [])
edges = live_graph.get("edges", [])
print(f"Dynamic Graph Nodes ({len(nodes)}):", [f"{n.get('id')}:{n.get('label')}" for n in nodes])
print(f"Dynamic Graph Edges ({len(edges)}):", [f"{e.get('source')}->{e.get('target')}" for e in edges])

# 4. Turn 3: Discuss clinical trial
t3 = post_json("http://localhost:8080/api/session/turn", {
    "session_id": sess_id,
    "utterance": "We discussed the SunRISe-1 clinical trial findings and the 77% complete response rate for INLEXZO"
})
print("\n--- Step 4: Rep introduces trial findings ---")
print("AI:", t3["bot_message"])

# 5. Turn 4: Catheter and procedure administration
t4 = post_json("http://localhost:8080/api/session/turn", {
    "session_id": sess_id,
    "utterance": "The clinic nurse had questions regarding the catheter insertion procedure and in-service training for INLEXZO"
})
print("\n--- Step 5: Rep switches to nurse catheter in-service ---")
print("AI:", t4["bot_message"])

# 6. Turn 5: Dysuria & safety
t5 = post_json("http://localhost:8080/api/session/turn", {
    "session_id": sess_id,
    "utterance": "Dr. Anurag raised questions about managing local urinary adverse events like dysuria and urgency"
})
print("\n--- Step 6: Rep switches to safety & dysuria ---")
print("AI:", t5["bot_message"])
