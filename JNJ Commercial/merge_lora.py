"""
Merge LoRA Weights with Base Model into a Standalone HuggingFace Directory
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def merge():
    lora_path = "models/slm_kg_trained_model"
    output_path = "models/slm_kg_merged"
    
    print(f"Reading LoRA configuration from: {lora_path}...")
    peft_cfg = PeftConfig.from_pretrained(lora_path)
    base_path = peft_cfg.base_model_name_or_path or "/home/mahendra/SLM_inference_time_new_data/llama_model_new_data"
    print(f"Base model path: {base_path}")
    
    print("Loading base model in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving standalone merged model to: {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("=" * 60)
    print(f"SUCCESS: Consolidated model saved to: {output_path}")
    print("Ready for vLLM & Transformers inference at maximum speed!")
    print("=" * 60)

if __name__ == "__main__":
    merge()
