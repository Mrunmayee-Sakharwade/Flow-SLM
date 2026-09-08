"""
Post-Training Inference & Verification Script
=============================================
Verifies the trained / merged SLM on sample Knowledge-Graph-conditioned
prompts across Oncology Specialist (OS) and Field Reimbursement Manager (FRM).

Usage:
    # 1. Test merged model:
    python test_inference.py --model_path models/slm_kg_merged_model

    # 2. Test base model with LoRA adapter directly:
    python test_inference.py --model_path meta-llama/Llama-3.2-1B-Instruct --lora_path models/slm_kg_lora_model
"""

import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


TEST_CASES = [
    {
        "name": "OS Case 1: Atlantic Urology Associates (Benefits Investigation & Coverage Barrier)",
        "role": "OS",
        "brand": "INLEXZO",
        "prompt": (
            "Brand: INLEXZO\n"
            "Persona: OS\n"
            "Contact: Dr. Robert Patel\n"
            "Account: Atlantic Urology Associates\n"
            "Known Account Friction: Coverage Policy & Benefits Investigation Barrier - Recent commercial and Medicare Advantage coverage changes have caused billing uncertainty; the office is rechecking patient out-of-pocket responsibility and waiting on benefits verification before committing to procedure scheduling.\n"
            "Target topic: treatment sequencing\n"
            "Conversation state: STATE_0_GREETING_INITIATION\n"
            "Allowed topics: account identification, clinical operational workflow, competitive landscape, cross functional collaboration, dosing administration, efficacy safety product info, relationship building, stakeholder management, territory engagement planning, treatment sequencing\n"
            "Forbidden topics: adverse events safety, off-label promotion, toxicity management\n"
            "Applicable responsibility: Discussing approved product information, including indications, clinical evidence, efficacy, safety within promotional scope, dosing, administration, patient identification, treatment positioning, and appropriate use of approved products.\n"
            "Compliance mandate: Always generate only a single follow-up question. Never refuse to capture call details. Do not repeat questions already asked.\n\n"
            "Conversation so far:\n"
            "AI: Hello, do you have a minute to capture the OS call notes for INLEXZO?\n"
            "User: I met Dr. Robert Patel at Atlantic Urology Associates. They are rechecking coverage criteria and waiting on benefits verification before scheduling."
        )
    },
    {
        "name": "OS Case 2: Northside Urology Group (Prior Authorization Clearance)",
        "role": "OS",
        "brand": "INLEXZO",
        "prompt": (
            "Brand: INLEXZO\n"
            "Persona: OS\n"
            "Contact: Dr. Sarah Jenkins\n"
            "Account: Northside Urology Group\n"
            "Known Account Friction: Prior Authorization Turnaround Barrier - Prior authorization submissions are currently in process with payers; clinic policy prohibits putting insertions on the procedure calendar until formal PA approval clears.\n"
            "Target topic: treatment sequencing\n"
            "Conversation state: STATE_2_PRIMARY_PURPOSE\n"
            "Allowed topics: account identification, clinical operational workflow, competitive landscape, cross functional collaboration, dosing administration, efficacy safety product info, relationship building, stakeholder management, territory engagement planning, treatment sequencing\n"
            "Forbidden topics: adverse events safety, off-label promotion, toxicity management\n"
            "Applicable responsibility: Understanding HCP treatment practices, prescribing behavior, clinical preferences, and educational needs related to approved products.\n"
            "Compliance mandate: Always generate only a single follow-up question. Never refuse to capture call details. Do not repeat questions already asked.\n\n"
            "Conversation so far:\n"
            "AI: Hello, do you have a minute to capture the OS call notes for INLEXZO?\n"
            "User: Dr. Sarah Jenkins at Northside Urology Group.\n"
            "AI: What was the main discussion regarding INLEXZO today?\n"
            "User: We discussed a BCG-unresponsive NMIBC patient. The PA is still in process and they cannot schedule until clearance."
        )
    },
    {
        "name": "OS Case 3: Capital Bladder Cancer Center (P&T Committee & Deductible)",
        "role": "OS",
        "brand": "INLEXZO",
        "prompt": (
            "Brand: INLEXZO\n"
            "Persona: OS\n"
            "Contact: Dr. Marcus Vance\n"
            "Account: Capital Bladder Cancer Center\n"
            "Known Account Friction: P&T Committee Review & Deductible Affordability Barrier - Institutional P&T committee review is currently pending for INLEXZO, and identified eligible patients are working through high deductible and payment timing, making affordability the practical obstacle.\n"
            "Target topic: territory engagement planning\n"
            "Conversation state: STATE_4_BARRIERS_LOGISTICS\n"
            "Allowed topics: account identification, clinical operational workflow, competitive landscape, cross functional collaboration, dosing administration, efficacy safety product info, relationship building, stakeholder management, territory engagement planning, treatment sequencing\n"
            "Forbidden topics: adverse events safety, off-label promotion, toxicity management\n"
            "Applicable responsibility: Planning compliant next steps, scheduling follow-ups, coordinating educational meetings, and aligning with field partners within scope.\n"
            "Compliance mandate: Always generate only a single follow-up question. Never refuse to capture call details. Do not repeat questions already asked.\n\n"
            "Conversation so far:\n"
            "AI: Hello, do you have a minute to capture the OS call notes for INLEXZO?\n"
            "User: I met with Dr. Marcus Vance at Capital Bladder Cancer Center.\n"
            "AI: What was the main discussion with Dr. Vance?\n"
            "User: We reviewed the P&T committee review status and patient deductible timing.\n"
            "AI: Did Dr. Vance provide an update on committee timing or patient affordability resources?\n"
            "User: Yes, the committee meets next week and they want assistance options for the patient deductible."
        )
    }
]

SYSTEM_PROMPT = (
    "Given the conversation history and Knowledge Graph context, generate the single most "
    "appropriate next question for the AI to ask the field representative.\n\n"
    "Requirements:\n"
    "- Ask exactly one question.\n"
    "- The question must follow naturally from the latest user response.\n"
    "- The question must focus on the target topic and adhere to the active J&J Core Responsibility.\n"
    "- Use only information available in the conversation history.\n"
    "- Do not repeat questions that have already been asked or ask for information already provided.\n"
    "- Do not answer the user.\n"
    "- Do not explain, summarize, advise, recommend, or suggest anything.\n"
    "- NEVER say 'I can't capture' or refuse to record call details.\n"
    "- NEVER generate compliance warnings or refusal messages.\n\n"
    "Output only the next question."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained SLM model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model directory or base model ID")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional path to LoRA adapter directory if not merged")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("  J&J COMMERCIAL ONCOLOGY SLM INFERENCE TEST SUITE")
    print(f"  Model Path: {args.model_path}")
    if args.lora_path:
        print(f"  LoRA Path : {args.lora_path}")
    print("=" * 75)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"\nLoading model ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    if args.lora_path:
        print(f"Applying LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()

    print("\nExecuting test queries across OS and FRM personas...\n")

    for i, test in enumerate(TEST_CASES, start=1):
        print("-" * 75)
        print(f"TEST {i}: {test['name']}")
        print(f"Role: {test['role']} | Brand: {test['brand']}")
        print("-" * 75)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test["prompt"]}
        ]

        formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print("Predicted Next Question:")
        print(f"  \"{response}\"")
        print(f"Latency: {elapsed_ms:.1f}ms | Tokens: {len(gen_tokens)} | Speed: {len(gen_tokens)/(elapsed_ms/1000):.1f} tok/s")
        print()


if __name__ == "__main__":
    main()
