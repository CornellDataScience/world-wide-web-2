"""
scripts/train_world_model.py

Supervised fine-tuning of the world model on NNetNav transition pairs.

This is a one-time offline step BEFORE agent RL training.
It trains the world model to predict (state_changes, next_obs) from
(current_obs, action) pairs extracted from NNetNav trajectories.

Run:
    cd dynaweb
    python scripts/train_world_model.py --data data/raw/train.jsonl

Output:
    checkpoints/world_model_epoch_1.pt   (stub mode: just logs pairs)
    checkpoints/world_model_final/       (real mode: HF model saved here)
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DynaWebConfig
from data.dataset import NNetNavDataset
from utils.prompt import format_wm_prompt_from_state_change


def parse_args():
    parser = argparse.ArgumentParser(description="Train DynaWeb world model")
    parser.add_argument("--data", type=str, default="data/raw/train.jsonl")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default="checkpoints/world_model_final")
    parser.add_argument("--no-stub", action="store_true")
    parser.add_argument("--export-pairs", action="store_true",
                        help="Export (prompt, target) pairs to JSONL for inspection")
    return parser.parse_args()


def export_training_pairs(dataset: NNetNavDataset, output_path: str):
    """
    Export all world model training pairs to a JSONL file.
    Each line: {"prompt": "...", "target": "..."}
    Useful for inspection and for feeding into external fine-tuning pipelines.

    Output file: data/raw/wm_training_pairs.jsonl
    """
    pairs = dataset.get_wm_training_pairs()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for sc_dict in pairs:
            # Build the prompt a world model would receive
            prompt = (
                f"Objective: (web navigation task)\n"
                f"Current action: {sc_dict['action']}\n\n"
                f"Current accessibility tree:\n{sc_dict['pre_obs']}\n\n"
                f"Predict state changes and next accessibility tree:"
            )
            # Build the target the world model should produce
            target = (
                f"[Web state changes]\n"
                f"Action '{sc_dict['action']}' was applied. "
                f"DOM changed: {sc_dict['dom_changed']}.\n\n"
                f"[Next page accessibility tree]\n"
                f"{sc_dict['post_obs']}"
            )
            f.write(json.dumps({"prompt": prompt, "target": target}) + "\n")

    print(f"[WM Training] Exported {len(pairs)} pairs to {output_path}")
    return pairs


def train_real_world_model(args, dataset: NNetNavDataset):
    """Fine-tune LLaMA as world model using HuggingFace Trainer."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    import torch
    from torch.utils.data import Dataset as TorchDataset

    print(f"Loading {args.model} for world model fine-tuning...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    # Build tokenized dataset from StateChange pairs
    pairs = dataset.get_wm_training_pairs()
    print(f"World model training pairs: {len(pairs)}")

    class WMDataset(TorchDataset):
        def __init__(self, pairs, tokenizer, max_len=2048):
            self.items = []
            for p in pairs:
                prompt = (
                    f"Action: {p['action']}\n"
                    f"Current page:\n{p['pre_obs']}\n\n"
                    f"Predict changes and next page:"
                )
                target = (
                    f"[Web state changes]\nApplied {p['action']}.\n\n"
                    f"[Next page accessibility tree]\n{p['post_obs']}"
                )
                full_text = prompt + "\n" + target + tokenizer.eos_token
                encoded = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                self.items.append({
                    "input_ids": encoded["input_ids"].squeeze(),
                    "attention_mask": encoded["attention_mask"].squeeze(),
                    "labels": encoded["input_ids"].squeeze().clone(),
                })

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    wm_dataset = WMDataset(pairs, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=wm_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting world model fine-tuning...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"World model saved to {args.output_dir}")


def main():
    args = parse_args()

    print("DynaWeb — World Model Training")
    print(f"Data: {args.data}")
    print(f"Stub mode: {not args.no_stub}")

    os.makedirs("checkpoints", exist_ok=True)

    # Load dataset
    dataset = NNetNavDataset(args.data)
    print(f"Loaded {len(dataset)} interactions")

    # Always export pairs for inspection
    pairs_path = "data/raw/wm_training_pairs.jsonl"
    pairs = export_training_pairs(dataset, pairs_path)

    if args.no_stub:
        train_real_world_model(args, dataset)
    else:
        # Stub mode: just show what would be trained on
        print(f"\n[Stub] Would fine-tune {args.model} on {len(pairs)} transition pairs.")
        print(f"[Stub] Sample training pair:")
        if pairs:
            p = pairs[0]
            print(f"  Action:   {p['action']}")
            print(f"  Pre obs:  {p['pre_obs'][:100]}...")
            print(f"  Post obs: {p['post_obs'][:100]}...")
            print(f"  DOM changed: {p['dom_changed']}")
        print(f"\n[Stub] Training pairs exported to {pairs_path}")
        print(f"[Stub] To train for real: python scripts/train_world_model.py --no-stub")


if __name__ == "__main__":
    main()