"""
Interactive Test Script: KG-Conditioned SLM Next-Question Predictor
===================================================================
User selects a role (FRM or OS), and the script runs a multi-turn
conversation where the SLM dynamically predicts the next question
using Knowledge Graph context retrieved at each turn.

Usage:
    python test.py --model-path outputs/merged_model

Requires:
    - Persona_Solid_Cancer_OS_FRM.json (KG data file)
    - engine/kg_context_retriever.py (KG context retriever)
    - A merged Llama-3.2-1B model at the specified path
"""

from __future__ import annotations

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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path so engine/ can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.kg_context_retriever import KGContextRetriever, FLAT_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Constants (matching infer.py)
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 768
MAX_NEW_TOKENS = 64

# Conversation state progression per role
STATE_FLOW = {
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


# ---------------------------------------------------------------------------
# Model loading (vLLM Engine)
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    """Loads the model using the high-throughput vLLM engine."""
    try:
        from vllm import LLM, SamplingParams
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

    gpu_mem = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.4"))
    print(f"\nLoading model with vLLM PagedAttention engine: {model_path} (GPU memory: {gpu_mem * 100:.0f}%)...")
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        max_model_len=MAX_SEQ_LENGTH,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1
    )

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
    print("vLLM engine loaded successfully.\n")
    return {"llm": llm, "sampling_params": sampling_params, "tokenizer": tokenizer}


# ---------------------------------------------------------------------------
# Generation (vLLM Engine)
# ---------------------------------------------------------------------------
def generate_next_question(model_bundle: dict, row: dict) -> str:
    """Builds prompt from row, runs vLLM model generation, returns predicted question."""
    tokenizer = model_bundle["tokenizer"]
    user_prompt = KGContextRetriever.build_flat_user_prompt(row)

    messages = [
        {"role": "system", "content": FLAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    llm = model_bundle["llm"]
    sampling_params = model_bundle["sampling_params"]
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Interactive conversation loop
# ---------------------------------------------------------------------------
def run_conversation(model_bundle: dict, kg: KGContextRetriever, role: str):
    """Runs an interactive multi-turn conversation with dynamic KG context."""
    role = role.upper()
    states = STATE_FLOW.get(role, STATE_FLOW["FRM"])
    state_idx = 0
    conversation_history = []
    covered_topics = set()

    print("=" * 60)
    print(f"  ROLE: {role} (Engine: vLLM PagedAttention)")
    print(f"  Starting interactive conversation...")
    print(f"  Type 'quit' to exit at any time.")
    print("=" * 60)

    # Generate the first question (cold start — no history yet)
    current_state = states[state_idx]

    row = kg.build_flat_row(
        role=role,
        current_state=current_state,
        brand="RYBREVANT",
        conversation_history=None,
        covered_topics=covered_topics,
    )

    first_question = generate_next_question(model_bundle, row)
    print(f"\nAI: {first_question}")
    conversation_history.append({"speaker": "AI", "text": first_question})

    consecutive_negatives = 0

    # Multi-turn loop
    while state_idx < len(states) - 1:
        # Get user input
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n[Session ended by user]")
            break
        if not user_input:
            continue

        # Check for continuous negativity / early exit
        is_neg = bool(re.match(r'^(no|nope|nah|none|nothing|stop|exit|cancel|no more|not really|done)[.!]?$', user_input, re.IGNORECASE))
        if is_neg:
            consecutive_negatives += 1
        else:
            consecutive_negatives = 0

        # Record user response
        conversation_history.append({"speaker": "User", "text": user_input})

        # If continuous negativity (>= 2) or explicit stop, trigger early exit guardrail
        if consecutive_negatives >= 2 or re.search(r'\b(stop|cancel|no more notes|exit|hangup)\b', user_input, re.IGNORECASE):
            closing_msg = "Understood. No further notes to capture for this visit. The call notes have been logged compliantly. Session closed."
            print(f"\n[Session Complete | Early Exit Guardrail]")
            print(f"AI: {closing_msg}")
            conversation_history.append({"speaker": "AI", "text": closing_msg})
            break

        # Advance state
        state_idx += 1
        if state_idx >= len(states):
            print("\nAI: Thank you, the call notes have been captured. Session closed.")
            break

        current_state = states[state_idx]

        # Build row with dynamic KG context + full conversation history
        row = kg.build_flat_row(
            role=role,
            current_state=current_state,
            brand="RYBREVANT",
            conversation_history=conversation_history,
            covered_topics=covered_topics,
        )

        # Show what KG context was retrieved (for debugging)
        print(f"\n  [State: {current_state}]")
        print(f"  [Target topic: {row['target_topic']}]")
        print(f"  [Allowed: {', '.join(row['kg_context']['allowed_topics'][:4])}...]")
        print(f"  [Forbidden: {', '.join(row['kg_context']['forbidden_topics'][:3])}...]")

        # Generate next question using the model
        next_question = generate_next_question(model_bundle, row)
        print(f"\nAI: {next_question}")

        # Record AI response in history
        conversation_history.append({"speaker": "AI", "text": next_question})

    # Print full transcript
    print("\n" + "=" * 60)
    print("  FULL CONVERSATION TRANSCRIPT")
    print("=" * 60)
    for entry in conversation_history:
        print(f"  {entry['speaker']}: {entry['text']}")
    print("=" * 60)


DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/home/mahendra/SLM_inference_time_new_data/llama_model_new_data")
if not os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists("outputs/merged_model"):
    DEFAULT_MODEL_PATH = "outputs/merged_model"


def main():
    parser = argparse.ArgumentParser(
        description="Interactive test: KG-conditioned SLM next-question prediction using vLLM"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the merged model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default="kg_os.json",
        help="Path to the Knowledge Graph JSON file",
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.model_path).exists():
        raise SystemExit(
            f"Model not found at '{args.model_path}'. "
            f"Run merge_adapter.py first or pass --model-path."
        )
    if not Path(args.kg_path).exists():
        raise SystemExit(
            f"KG file not found at '{args.kg_path}'. "
            f"Pass --kg-path pointing at kg_os.json."
        )

    # Load KG and Model
    print("Loading Knowledge Graph...")
    kg = KGContextRetriever(kg_json_path=args.kg_path)

    model_bundle = load_model(args.model_path)

    # Ask user to pick a role
    print("=" * 60)
    print("  J&J Commercial Oncology — SLM Next-Question Predictor (vLLM)")
    print("=" * 60)
    print("\nSelect a role:")
    print("  1. FRM (Field Reimbursement Manager)")
    print("  2. OS  (Oncology Sales Representative)")

    choice = input("\nEnter 1 or 2: ").strip()
    if choice == "1":
        role = "FRM"
    elif choice == "2":
        role = "OS"
    else:
        print(f"Invalid choice '{choice}'. Defaulting to FRM.")
        role = "FRM"

    # Run the conversation
    run_conversation(model_bundle, kg, role)


if __name__ == "__main__":
    main()
