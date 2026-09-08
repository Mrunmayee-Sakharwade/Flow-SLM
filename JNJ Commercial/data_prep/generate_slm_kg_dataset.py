"""
KG-Conditioned SLM Training Dataset Generator
=============================================
Transforms the 23,811 historical transcript turns into an instruction-tuned,
Knowledge-Graph-grounded training dataset for Small Language Models (SLMs).

Outputs:
  - data_prep/slm_train.jsonl (80% Training split - ChatML format)
  - data_prep/slm_val.jsonl   (10% Validation split - ChatML format)
  - data_prep/slm_test.jsonl  (10% Test split - ChatML format)
  - data_prep/slm_alpaca_dataset.json (Alpaca Instruction-Input-Output format)
  - data_prep/slm_dataset_summary.json (Dataset metadata & token statistics)
"""

import sys
import os
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Any

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from engine.kg_context_retriever import KGContextRetriever, FLAT_SYSTEM_PROMPT

def main():
    print("=" * 75)
    print("STARTING KG-AUGMENTED SLM DATASET GENERATION PIPELINE")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data_prep", "training_turns_dataset.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found at {dataset_path}")
        
    print(f"Loading raw turns dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_turns = json.load(f)
        
    print(f"Total raw turns loaded: {len(raw_turns):,}")
    
    # Initialize KG Context Retriever with fast graph indexing (no neural embedder needed for batch compilation)
    kg_retriever = KGContextRetriever(init_embedder=False)
    
    slm_samples = []
    alpaca_samples = []
    
    role_counts = Counter()
    state_counts = Counter()
    target_topic_counts = Counter()
    
    # Group turns by dialogue_id to guarantee 100% dialogue integrity and sequential continuity
    dialogues_map = defaultdict(list)
    for turn in raw_turns:
        d_id = turn.get("dialogue_id")
        dialogues_map[d_id].append(turn)
        
    print(f"Total unique dialogues loaded: {len(dialogues_map):,}")

    # Ensure all turns in every dialogue are strictly sorted by turn_index
    for d_id in dialogues_map:
        dialogues_map[d_id].sort(key=lambda t: t.get("turn_index", 0))

    # Build SLM samples per dialogue
    dialogue_samples_map = {}
    role_counts = Counter()
    state_counts = Counter()
    target_topic_counts = Counter()
    total_turns_processed = 0

    for d_id, turns in dialogues_map.items():
        d_samples = []
        d_alpaca = []
        covered_topics_so_far = set()
        # Build cumulative conversation history turn by turn
        cumulative_history = []

        for turn in turns:
            role = turn.get("role", "OS")
            state = turn.get("state", "STATE_0_GREETING_INITIATION")
            curr_q = turn.get("current_question", "")
            cand_ans = turn.get("candidate_answer", "")
            brand = turn.get("brand", "INLEXZO")
            topics = turn.get("kg_topics", [])
            is_term = turn.get("is_terminal", False)
            next_q = turn.get("next_question")
            persona_name = turn.get("rep_name", "")

            for t in topics:
                covered_topics_so_far.add(t)

            # Build the flat row using dynamic KG context retriever
            flat_row = kg_retriever.build_flat_row(
                role=role,
                current_state=state,
                brand=brand,
                persona_name=persona_name,
                conversation_history=cumulative_history if cumulative_history else None,
                covered_topics=covered_topics_so_far,
                detected_topics=topics,
                candidate_answer=cand_ans
            )

            target_topic = flat_row["target_topic"]

            # Build flat user prompt (same format as infer.py)
            user_prompt = KGContextRetriever.build_flat_user_prompt(flat_row)

            # Ground-truth output: next question or compliant session wrap-up if terminal turn
            if next_q and next_q.strip():
                target_output = next_q.strip()
            else:
                target_output = "Thank you, the call notes have been captured and logged compliantly. Session closed."

            # ChatML Format Sample (flat prompt, matching infer.py)
            chatml_sample = {
                "turn_id": turn.get("turn_id"),
                "dialogue_id": d_id,
                "role": role,
                "state": state,
                "target_topic": target_topic,
                "messages": [
                    {"role": "system", "content": FLAT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target_output}
                ],
                "kg_context_summary": {
                    "target_topic": target_topic,
                    "relevant_duties": [flat_row["kg_context"]["applicable_responsibility"]] if flat_row["kg_context"]["applicable_responsibility"] else [],
                    "allowed_topics": flat_row["kg_context"]["allowed_topics"],
                    "forbidden_topics": flat_row["kg_context"]["forbidden_topics"]
                },
                "flat_row": flat_row
            }

            # Alpaca Format Sample (flat prompt)
            alpaca_sample = {
                "instruction": FLAT_SYSTEM_PROMPT,
                "input": user_prompt,
                "output": target_output,
                "metadata": {
                    "turn_id": turn.get("turn_id"),
                    "role": role,
                    "state": state,
                    "target_topic": target_topic
                }
            }

            d_samples.append(chatml_sample)
            d_alpaca.append(alpaca_sample)
            role_counts[role] += 1
            state_counts[state] += 1
            target_topic_counts[target_topic] += 1
            total_turns_processed += 1

            # Accumulate conversation history for next turn
            cumulative_history.append({"speaker": "AI", "text": curr_q})
            cumulative_history.append({"speaker": "User", "text": cand_ans})

        dialogue_samples_map[d_id] = {
            "chatml": d_samples,
            "alpaca": d_alpaca,
            "role": turns[0].get("role", "OS")
        }

    print(f"Constructed {total_turns_processed:,} KG-augmented training turns across {len(dialogue_samples_map):,} complete dialogues.")

    # Sort strictly by Serial Number (s_no 1 to 1498)
    os_dialogue_ids = sorted(
        list(dialogue_samples_map.keys()),
        key=lambda x: int(x.split("_")[-1])
    )

    n_total = len(os_dialogue_ids)
    # Match Dhananjay's original partition (1,196 train, 148 val, 150+ test)
    n_train = 1196
    n_val = 148
    
    train_ids = os_dialogue_ids[:n_train]
    val_ids = os_dialogue_ids[n_train:n_train + n_val]
    test_ids = os_dialogue_ids[n_train + n_val:]

    print(f"OS Dialogues: {n_total:,} (S.No {int(os_dialogue_ids[0].split('_')[-1])} to {int(os_dialogue_ids[-1].split('_')[-1])})")
    print(f"\nSerial-Number-Wise Partition Summary (OS_DHAN Partition: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}):")
    print(f"  - Train Split (S.No {int(train_ids[0].split('_')[-1]):04d} - {int(train_ids[-1].split('_')[-1]):04d}): {len(train_ids)} dialogues")
    print(f"  - Val Split   (S.No {int(val_ids[0].split('_')[-1]):04d} - {int(val_ids[-1].split('_')[-1]):04d}): {len(val_ids)} dialogues")
    print(f"  - Test Split  (S.No {int(test_ids[0].split('_')[-1]):04d} - {int(test_ids[-1].split('_')[-1]):04d}): {len(test_ids)} dialogues")


    # Flatten samples in strict chronological order per dialogue
    def flatten_dialogues(ids_list):
        chatml_out = []
        alpaca_out = []
        for d_id in ids_list:
            chatml_out.extend(dialogue_samples_map[d_id]["chatml"])
            alpaca_out.extend(dialogue_samples_map[d_id]["alpaca"])
        return chatml_out, alpaca_out

    train_chatml, train_alpaca = flatten_dialogues(train_ids)
    val_chatml, val_alpaca = flatten_dialogues(val_ids)
    test_chatml, test_alpaca = flatten_dialogues(test_ids)

    all_alpaca = train_alpaca + val_alpaca + test_alpaca

    print(f"Turns Split (100% In-Sequence):")
    print(f"  - Training Set:   {len(train_chatml):,} turns across {len(train_ids)} complete dialogues ({len(train_chatml)/total_turns_processed*100:.1f}%)")
    print(f"  - Validation Set: {len(val_chatml):,} turns across {len(val_ids)} complete dialogues ({len(val_chatml)/total_turns_processed*100:.1f}%)")
    print(f"  - Test Set:       {len(test_chatml):,} turns across {len(test_ids)} complete dialogues ({len(test_chatml)/total_turns_processed*100:.1f}%)")
    print(f"  - Total Turns:    {len(train_chatml) + len(val_chatml) + len(test_chatml):,} (100% of turns preserved)")

    # Write out JSONL files
    data_prep_dir = os.path.join(base_dir, "data_prep")
    
    train_file = os.path.join(data_prep_dir, "slm_train.jsonl")
    val_file = os.path.join(data_prep_dir, "slm_val.jsonl")
    test_file = os.path.join(data_prep_dir, "slm_test.jsonl")
    alpaca_file = os.path.join(data_prep_dir, "slm_alpaca_dataset.json")
    summary_file = os.path.join(data_prep_dir, "slm_dataset_summary.json")
    
    def write_jsonl(filepath, data_list):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    write_jsonl(train_file, train_chatml)
    write_jsonl(val_file, val_chatml)
    write_jsonl(test_file, test_chatml)
    
    with open(alpaca_file, "w", encoding="utf-8") as f:
        json.dump(all_alpaca, f, indent=2, ensure_ascii=False)
        
    summary_data = {
        "total_slm_samples": total_turns_processed,
        "total_dialogues": len(dialogue_samples_map),
        "splits": {
            "train_dialogues": len(train_ids),
            "train_turns": len(train_chatml),
            "val_dialogues": len(val_ids),
            "val_turns": len(val_chatml),
            "test_dialogues": len(test_ids),
            "test_turns": len(test_chatml)
        },
        "role_distribution": dict(role_counts),
        "state_distribution": dict(state_counts.most_common()),
        "target_topics_distribution": dict(target_topic_counts.most_common()),
        "file_paths": {
            "train": "data_prep/slm_train.jsonl",
            "val": "data_prep/slm_val.jsonl",
            "test": "data_prep/slm_test.jsonl",
            "alpaca": "data_prep/slm_alpaca_dataset.json"
        }
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    print("\n" + "=" * 75)
    print("SLM KG-AUGMENTED DATASET CREATION COMPLETED SUCCESSFULLY!")
    print(f"Generated Training File:   {train_file}")
    print(f"Generated Validation File: {val_file}")
    print(f"Generated Test File:       {test_file}")
    print(f"Generated Summary File:    {summary_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()

