"""
J&J Commercial Oncology SLM Fine-Tuning Script (Llama-3.2-1B)
=============================================================
Fine-tunes Llama-3.2-1B on the Knowledge-Graph-Conditioned Next-Question
Prediction dataset using PEFT / LoRA (or QLoRA).

Usage:
  # Standard 16-bit LoRA (Recommended for 16GB+ VRAM, e.g., RTX 3090/4090, A10G, A100):
  python train_slm.py --base_model meta-llama/Llama-3.2-1B-Instruct \
                      --train_file data/slm_train.jsonl \
                      --val_file data/slm_val.jsonl \
                      --output_dir models/slm_kg_lora_model \
                      --epochs 2 \
                      --batch_size 4 \
                      --grad_accum 4

  # 4-bit QLoRA (For consumer GPUs with 8GB-12GB VRAM):
  python train_slm.py --base_model meta-llama/Llama-3.2-1B-Instruct \
                      --train_file data/slm_train.jsonl \
                      --val_file data/slm_val.jsonl \
                      --output_dir models/slm_kg_lora_model \
                      --use_4bit
"""

import os
import sys
import json
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B on J&J KG dataset")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--train_file", type=str, default="data/slm_train.jsonl",
                        help="Path to slm_train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/slm_val.jsonl",
                        help="Path to slm_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/slm_kg_lora_model",
                        help="Output directory for trained LoRA adapter")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for AdamW")
    parser.add_argument("--max_seq_len", type=int, default=768,
                        help="Maximum sequence length (prompt + answer)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Enable 4-bit QLoRA quantization via bitsandbytes")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Optional limit on train samples for quick testing")
    return parser.parse_args()


def load_jsonl(filepath: str, limit: int = None):
    """Loads a JSONL dataset file into a list of dicts."""
    if not os.path.exists(filepath):
        # Fallback to ../data_prep if in subfolder
        parent_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filepath)
        if os.path.exists(parent_candidate):
            filepath = parent_candidate
        else:
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print(f"Loading data from: {filepath}")
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if limit and len(samples) >= limit:
                    break
    print(f"Loaded {len(samples):,} samples from {os.path.basename(filepath)}")
    return samples


def main():
    args = parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"\n[Error] Missing required training dependencies: {e}")
        print("Please run: pip install -r requirements.txt\n")
        sys.exit(1)

    print("=" * 75)
    print("  J&J COMMERCIAL ONCOLOGY SLM FINE-TUNING PIPELINE")
    print("  Base Model :", args.base_model)
    print("  LoRA Config: r =", args.lora_r, "| alpha =", args.lora_alpha, "| 4-bit =", args.use_4bit)
    print("  Epochs     :", args.epochs)
    print("  Batch Size :", args.batch_size, f"(Grad Accum: {args.grad_accum} -> Effective: {args.batch_size * args.grad_accum})")
    print("  Learning Rate:", args.lr)
    print("=" * 75)

    # 1. Load Raw Datasets
    train_raw = load_jsonl(args.train_file, limit=args.limit_samples)
    val_raw = load_jsonl(args.val_file, limit=200)

    # 2. Tokenizer Setup
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Format with Chat Template
    print("Applying chat template to conversation messages...")
    train_texts = [{"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)} for x in train_raw]
    val_texts = [{"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)} for x in val_raw]

    train_ds = Dataset.from_list(train_texts)
    val_ds = Dataset.from_list(val_texts)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False
        )

    print("Tokenizing datasets...")
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 4. Model Loading (Full 16-bit or 4-bit QLoRA)
    print("\nLoading base model weights...")
    device_map = "auto" if torch.cuda.is_available() else None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if args.use_4bit:
        print("Configuring 4-bit NormalFloat4 (NF4) quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

    # 5. LoRA Configuration
    print("\nApplying LoRA Adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 6. Training Arguments & Trainer Setup
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator
    )

    print("\n" + "=" * 75)
    print("STARTING TRAINING LOOP")
    print("=" * 75)
    train_result = trainer.train()

    print("\nTraining completed successfully!")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    # 7. Save LoRA Adapter & Tokenizer
    print(f"\nSaving fine-tuned LoRA adapter to: {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    meta = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "final_train_loss": train_result.training_loss
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 75)
    print(f"TRAINING COMPLETE! LoRA Adapter saved to: {args.output_dir}")
    print(f"Next Step: Run 'python merge_lora.py --lora_path {args.output_dir}' to merge weights.")
    print("=" * 75)


if __name__ == "__main__":
    main()
