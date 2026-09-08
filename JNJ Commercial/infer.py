"""
Multi-Turn Next-Question Inference Engine using vLLM
=====================================================
Executes full multi-turn conversational inference for the fine-tuned
merged Llama-3.2-1B model using high-throughput vLLM (PagedAttention).

At each turn of the conversation:
  1. The user provides a response.
  2. The cumulative transcript ("AI: ...\nUser: ...") is accumulated.
  3. Dynamic Knowledge Graph context (allowed topics, forbidden topics,
     applicable responsibility, compliance rules) is retrieved for the current state.
  4. The flat prompt is assembled.
  5. vLLM generates the next question.

Usage:
    # 1. Interactive Multi-Turn Mode (select FRM or OS and chat turn-by-turn):
    python infer.py --model-path outputs/merged_model

    # 2. Run with a specific role directly:
    python infer.py --model-path outputs/merged_model --role FRM
    python infer.py --model-path outputs/merged_model --role OS

    # 3. Process a custom multi-turn dialogue JSON file:
    python infer.py --model-path outputs/merged_model --dialogue-file my_dialogue.json

Requires:
    pip install vllm
"""

import argparse
import json
import os
import re
import sys

# Default to GPU 1 if not specified
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use stable v0 engine to avoid JIT sampler compilation
os.environ.setdefault("VLLM_USE_V1", "0")

from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.kg_context_retriever import KGContextRetriever, FLAT_SYSTEM_PROMPT

MAX_SEQ_LENGTH = 768
MAX_NEW_TOKENS = 64

# Canonical multi-turn conversation state sequence per role
ROLE_STATE_FLOW = {
    "FRM": [
        "STATE_0_GREETING_INITIATION",
        "STATE_1_ACCOUNT_STAKEHOLDER",
        "STATE_2_PRIMARY_PURPOSE",
        "STATE_3A_PRIOR_AUTH_PAYER",
        "STATE_3B_AFFORDABILITY_COPAY_PAP",
        "STATE_3C_HUB_SPECIALTY_PHARMACY",
        "STATE_6_NEXT_ACTIONS",
        "STATE_7_WRAP_UP_CONFIRMATION",
    ],
    "OS": [
        "STATE_0_GREETING_INITIATION",
        "STATE_1_ACCOUNT_STAKEHOLDER",
        "STATE_2_PRIMARY_PURPOSE",
        "STATE_3E_WORKFLOW_DEMO_REFRESHER",
        "STATE_4_BARRIERS_LOGISTICS",
        "STATE_5_CROSS_FUNCTIONAL_HANDOFF",
        "STATE_6_NEXT_ACTIONS",
        "STATE_7_WRAP_UP_CONFIRMATION",
    ],
}


def load_vllm_model(model_path: str, gpu_memory_utilization: float = 0.2) -> dict:
    """Initializes the vLLM engine with PagedAttention."""
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "\n[Error] 'vllm' is required for inference but is not installed.\n"
            "Please install it using:\n"
            "    pip install vllm\n"
        )

    # Check if tokenizer_config.json contains the invalid class 'TokenizersBackend'
    tok_cfg_path = os.path.join(model_path, "tokenizer_config.json")
    tokenizer_name = model_path
    if os.path.exists(tok_cfg_path):
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
                try:
                    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
                    with open(tok_cfg_path, "w", encoding="utf-8") as f:
                        json.dump(tok_cfg, f, indent=2)
                except Exception:
                    tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
        except Exception:
            pass

    print(f"\nInitializing vLLM Engine from: {model_path} (GPU memory: {gpu_memory_utilization * 100:.0f}%)...")
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        max_model_len=MAX_SEQ_LENGTH,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1
    )
    
    # Retrieve tokenizer directly from the loaded vLLM engine
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
            
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS
    )
    print("vLLM Engine initialized successfully.\n")
    return {"llm": llm, "sampling_params": sampling_params, "tokenizer": tokenizer}


def predict_next_question_turn(
    vllm_bundle: dict,
    kg: KGContextRetriever,
    role: str,
    current_state: str,
    conversation_history: Optional[List[Dict[str, str]]],
    brand: Optional[str] = None,
    persona_name: Optional[str] = None,
    covered_topics: Optional[set] = None
) -> Dict[str, Any]:
    """
    Executes a single turn prediction in the multi-turn sequence:
      1. Retrieves dynamic KG context for the active role & state.
      2. Injects cumulative conversation history.
      3. Uses vLLM to predict the next question.
    """
    # Build dynamic flat row from Knowledge Graph
    row = kg.build_flat_row(
        role=role,
        current_state=current_state,
        brand=brand,
        persona_name=persona_name,
        conversation_history=conversation_history,
        covered_topics=covered_topics
    )

    # Format into flat prompt
    user_prompt = KGContextRetriever.build_flat_user_prompt(row)

    messages = [
        {"role": "system", "content": FLAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # Format chat template
    tokenizer = vllm_bundle["tokenizer"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Run vLLM generation
    llm = vllm_bundle["llm"]
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    predicted_q = outputs[0].outputs[0].text.strip()

    # Detailed Knowledge Graph Context Logging
    print("\n" + "═" * 75)
    print(f" 📊 KG CONTEXT & SLM INFERENCE LOG | State: {current_state}")
    print("═" * 75)
    print(f" ▶ Active Role       : {role}")
    print(f" ▶ Brand Focus       : {brand or 'RYBREVANT'}")
    print(f" ▶ Target KG Topic   : {row.get('target_topic', 'N/A')}")
    print(f" ▶ Unaddressed Topics: {row.get('unaddressed_topics', 'None')}")
    print(f" ▶ In-Scope Duties   :\n   {row.get('relevant_duties', 'None')}")
    print("\n ─── [FULL INJECTED KG CONTEXT PROMPT] ───")
    for line in user_prompt.split("\n"):
        print(f"   {line}")
    print(" ─────────────────────────────────────────")
    print(f" ▶ SLM Predicted Next Question:\n   \"{predicted_q}\"")
    print("═" * 75 + "\n")

    return {
        "predicted_question": predicted_q,
        "row": row,
        "user_prompt": user_prompt,
        "target_topic": row.get("target_topic", "")
    }


def run_interactive_multi_turn(
    vllm_bundle: dict,
    kg: KGContextRetriever,
    role: str,
    brand: str = "RYBREVANT",
    persona_name: Optional[str] = None
):
    """
    Runs an interactive multi-turn interview session.
    At every turn, user input is captured, cumulative history is built,
    and vLLM predicts the next question based on dynamic KG context.
    """
    role = role.upper()
    states = ROLE_STATE_FLOW.get(role, ROLE_STATE_FLOW["FRM"])
    conversation_history: List[Dict[str, str]] = []
    covered_topics: set = set()

    print("=" * 70)
    print(f"  J&J MULTI-TURN SLM INFERENCE (vLLM Engine)")
    print(f"  Active Role : {role}")
    print(f"  Brand Focus : {brand}")
    print(f"  State Flow  : {len(states)} sequential states")
    print(f"  Type 'quit' at any prompt to exit.")
    print("=" * 70)

    # Turn 0: Generate opening question (cold-start)
    turn_idx = 0
    current_state = states[turn_idx]
    
    result = predict_next_question_turn(
        vllm_bundle=vllm_bundle,
        kg=kg,
        role=role,
        current_state=current_state,
        conversation_history=None,
        brand=brand,
        persona_name=persona_name,
        covered_topics=covered_topics
    )
    
    opening_q = result["predicted_question"]
    print(f"\n[Turn {turn_idx + 1}/{len(states)} | State: {current_state}]")
    print(f"AI: {opening_q}")
    conversation_history.append({"speaker": "AI", "text": opening_q})

    consecutive_negatives = 0

    # Multi-turn interaction loop
    while turn_idx < len(states) - 1:
        # 1. Capture user response
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n[Session ended by user]")
            break
        if not user_input:
            continue

        # Check for continuous negativity guardrail
        is_neg = bool(re.match(r'^(no|nope|nah|none|nothing|stop|exit|cancel|no more|not really|done)[.!]?$', user_input, re.IGNORECASE))
        if is_neg:
            consecutive_negatives += 1
        else:
            consecutive_negatives = 0

        conversation_history.append({"speaker": "User", "text": user_input})

        # If continuous negativity (>= 2) or explicit stop, trigger early exit guardrail
        if consecutive_negatives >= 2 or re.search(r'\b(stop|cancel|no more notes|exit|hangup)\b', user_input, re.IGNORECASE):
            closing_msg = "Understood. No further notes to capture for this visit. The call notes have been logged compliantly. Session closed."
            print(f"\n[Session Complete | Early Exit Guardrail]")
            print(f"AI: {closing_msg}")
            conversation_history.append({"speaker": "AI", "text": closing_msg})
            break

        # 2. Advance to next state
        turn_idx += 1
        current_state = states[turn_idx]

        # 3. Predict next question with vLLM using cumulative history + dynamic KG
        result = predict_next_question_turn(
            vllm_bundle=vllm_bundle,
            kg=kg,
            role=role,
            current_state=current_state,
            conversation_history=conversation_history,
            brand=brand,
            persona_name=persona_name,
            covered_topics=covered_topics
        )

        next_q = result["predicted_question"]
        print(f"\n[Turn {turn_idx + 1}/{len(states)} | State: {current_state} | Target: {result['target_topic']}]")
        print(f"AI: {next_q}")
        conversation_history.append({"speaker": "AI", "text": next_q})

    # Display final cumulative transcript summary
    print("\n" + "=" * 70)
    print("  FINAL MULTI-TURN CONVERSATION TRANSCRIPT")
    print("=" * 70)
    for entry in conversation_history:
        print(f"  {entry['speaker']}: {entry['text']}")
    print("=" * 70 + "\n")


def run_dialogue_file_multi_turn(
    vllm_bundle: dict,
    kg: KGContextRetriever,
    dialogue_file: str
):
    """
    Processes a pre-recorded multi-turn dialogue JSON file turn-by-turn through vLLM.
    """
    with open(dialogue_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle single turn dict or list of turns
    turns = data if isinstance(data, list) else [data]
    print(f"\nProcessing {len(turns)} dialogue turns through vLLM...")

    for i, turn in enumerate(turns):
        role = turn.get("role", "FRM")
        state = turn.get("state", "STATE_0_GREETING_INITIATION")
        brand = turn.get("brand", "RYBREVANT")
        context_str = turn.get("context")

        history_list = None
        if context_str:
            history_list = []
            for line in context_str.split("\n"):
                line = line.strip()
                if line.startswith("AI:"):
                    history_list.append({"speaker": "AI", "text": line.split(":", 1)[1].strip()})
                elif line.startswith("User:"):
                    history_list.append({"speaker": "User", "text": line.split(":", 1)[1].strip()})

        result = predict_next_question_turn(
            vllm_bundle=vllm_bundle,
            kg=kg,
            role=role,
            current_state=state,
            conversation_history=history_list,
            brand=brand,
            persona_name=turn.get("persona_name")
        )

        print(f"\n--- Turn {i + 1} ({role} | {state}) ---")
        print(f"Predicted Next Question: {result['predicted_question']}")


DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/home/mahendra/SLM_inference_time_new_data/llama_model_new_data")
if not os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists("outputs/merged_model"):
    DEFAULT_MODEL_PATH = "outputs/merged_model"


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Turn SLM Next-Question Inference Engine using vLLM"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the merged Llama-3.2 model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default="Persona_Solid_Cancer_OS_FRM.json",
        help="Path to the Knowledge Graph JSON file",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["FRM", "OS"],
        default=None,
        help="Field role ('FRM' or 'OS'). If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="RYBREVANT",
        help="Oncology product brand focus (default: RYBREVANT)",
    )
    parser.add_argument(
        "--dialogue-file",
        type=str,
        default=None,
        help="Optional path to a JSON file containing pre-defined dialogue turn(s)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.4")),
        help="Fraction of GPU memory to allocate for vLLM (default: 0.4)",
    )
    args = parser.parse_args()

    # Validate model path
    if not Path(args.model_path).exists():
        raise SystemExit(
            f"[Error] Model directory '{args.model_path}' not found.\n"
            f"Please verify your merged model path."
        )

    # Initialize Knowledge Graph and vLLM
    print("Loading Knowledge Graph context retriever...")
    kg = KGContextRetriever(kg_json_path=args.kg_path)

    vllm_bundle = load_vllm_model(args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)

    # If dialogue file is provided, process it
    if args.dialogue_file:
        run_dialogue_file_multi_turn(vllm_bundle, kg, args.dialogue_file)
        return

    # Interactive Multi-Turn Mode
    selected_role = args.role
    if not selected_role:
        print("=" * 60)
        print("  Select Field Role for Multi-Turn Session:")
        print("    1. FRM (Field Reimbursement Manager)")
        print("    2. OS  (Oncology Sales Representative)")
        print("=" * 60)
        choice = input("Enter 1 or 2: ").strip()
        selected_role = "FRM" if choice == "1" else "OS"

    run_interactive_multi_turn(
        vllm_bundle=vllm_bundle,
        kg=kg,
        role=selected_role,
        brand=args.brand
    )


if __name__ == "__main__":
    main()
