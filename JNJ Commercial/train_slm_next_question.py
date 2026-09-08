import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SLM on dynamic KG dataset")
    parser.add_argument("--model_name", default="/home/mahendra/SLM_inference_time_new_data/llama_model_new_data")
    parser.add_argument("--train_file", default="data_prep/slm_train.jsonl")
    parser.add_argument("--val_file", default="data_prep/slm_val.jsonl")
    parser.add_argument("--output_dir", default="models/slm_kg_trained_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=768)
    return parser.parse_args()

def load_data(filepath, limit=None):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if limit and len(data) >= limit:
                    break
    return data

def main():
    args = parse_args()
    print("=" * 70)
    print("INITIALIZING J&J COMMERCIAL ONCOLOGY SLM FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Base Model: {args.model_name}")
    print(f"Train File: {args.train_file}")
    print(f"Val File:   {args.val_file}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum})")

    train_raw = load_data(args.train_file)
    val_raw = load_data(args.val_file, 200)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting datasets with chat template...")
    train_ds = Dataset.from_list([{"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)} for x in train_raw])
    val_ds = Dataset.from_list([{"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)} for x in val_raw])
    print(f"Train samples: {len(train_ds):,} | Validation samples: {len(val_ds):,}")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_len)

    print("Tokenizing datasets...")
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )

    print("Applying LoRA adapters...")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, peft_cfg)
    model.print_trainable_parameters()

    # DataCollatorForLanguageModeling automatically sets labels = input_ids for CausalLM loss computation
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=(dtype == torch.float16 and torch.cuda.is_available()),
        bf16=(dtype == torch.bfloat16 and torch.cuda.is_available()),
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        args=training_args,
        data_collator=data_collator
    )

    print("\n" + "=" * 70)
    print("STARTING SLM TRAINING ON DYNAMIC KNOWLEDGE GRAPH DATASET...")
    print("=" * 70)
    trainer.train()

    print("\nSaving fine-tuned model and tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SUCCESS: Model saved to {args.output_dir}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
