"""
Consolidate & Merge LoRA Adapter into Standalone Hugging Face Model
==================================================================
Merges LoRA weight matrices into base model weights and exports a full,
unquantized float16 model directory ready for fast inference (vLLM, Transformers, Ollama).

Usage:
    python merge_lora.py --base_model meta-llama/Llama-3.2-1B-Instruct \
                         --lora_path models/slm_kg_lora_model \
                         --output_dir models/slm_kg_merged_model
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path or HuggingFace ID (defaults to adapter config base)")
    parser.add_argument("--lora_path", type=str, default="models/slm_kg_lora_model",
                        help="Directory containing trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="models/slm_kg_merged_model",
                        help="Destination directory for merged standalone model")
    return parser.parse_args()


def merge():
    args = parse_args()

    print("=" * 75)
    print("  J&J COMMERCIAL ONCOLOGY SLM WEIGHT MERGER")
    print(f"  LoRA Adapter Path : {args.lora_path}")
    print(f"  Merged Output Dir : {args.output_dir}")
    print("=" * 75)

    if not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {args.lora_path}")

    print("\nReading LoRA configuration...")
    peft_config = PeftConfig.from_pretrained(args.lora_path)
    base_model_path = args.base_model or peft_config.base_model_name_or_path or "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Base Model: {base_model_path}")

    # Determine optimal precision
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading base model in {dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging LoRA weights with base model layers (zero inference overhead)...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=" * 75)
    print(f"SUCCESS: Consolidated model saved to: {args.output_dir}")
    print("Ready for deployment with vLLM, HuggingFace pipeline, or Ollama!")
    print("=" * 75)


if __name__ == "__main__":
    merge()
