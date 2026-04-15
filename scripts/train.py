"""
scripts/train.py

Main entry point for DynaWeb training.

Run in stub mode (no GPU, no model weights needed):
    cd dynaweb
    python scripts/train.py

Run with real LLaMA:
    cd dynaweb
    python scripts/train.py --no-stub --agent-model meta-llama/Llama-3.1-8B-Instruct

With real data:
    python scripts/train.py --train-data data/raw/train.jsonl --val-data data/raw/val.jsonl
"""

import sys
import os
import argparse

# Make sure project root is on path regardless of where script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DynaWebConfig
from data.dataset import NNetNavDataset
from models.agent import AgentPolicy
from models.world_model import WebWorldModel
from models.reward import SelfAssessReward
from training.trainer import DynaWebTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DynaWeb agent")

    parser.add_argument("--no-stub", action="store_true",
                        help="Use real LLaMA models instead of stubs")
    parser.add_argument("--agent-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path for agent")
    parser.add_argument("--wm-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path for world model")
    parser.add_argument("--train-data", type=str,
                        default="data/raw/train.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--val-data", type=str,
                        default="data/raw/val.jsonl",
                        help="Path to validation JSONL file")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rollout-n", type=int, default=2,
                        help="Rollouts per task (paper uses 8)")
    parser.add_argument("--dream-length", type=int, default=3,
                        help="Max steps per imagined trajectory (paper uses 5)")
    parser.add_argument("--real-ratio", type=float, default=0.5,
                        help="Fraction of batch replaced with real trajectories")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")

    return parser.parse_args()


def build_config(args) -> DynaWebConfig:
    return DynaWebConfig(
        use_stub_models=not args.no_stub,
        agent_model_name=args.agent_model,
        wm_model_name=args.wm_model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        epochs=args.epochs,
        train_batch_size=args.batch_size,
        rollout_n=args.rollout_n,
        dream_length=args.dream_length,
        real_traj_ratio=args.real_ratio,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


def main():
    args = parse_args()
    config = build_config(args)

    print(f"\nDynaWeb — Model-Based RL for Web Agents")
    print(f"Stub mode: {config.use_stub_models}")
    if not config.use_stub_models:
        print(f"Agent model: {config.agent_model_name}")
        print(f"World model: {config.wm_model_name}")

    # ------------------------------------------------------------------
    # 1. Load tokenizer (only needed for real models)
    # ------------------------------------------------------------------
    tokenizer = None
    if not config.use_stub_models:
        from transformers import AutoTokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.agent_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 2. Load datasets
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = NNetNavDataset(config.train_data_path)
    val_dataset = NNetNavDataset(config.val_data_path)
    print(f"  Train: {len(train_dataset)} interactions")
    print(f"  Val:   {len(val_dataset)} interactions")

    # ------------------------------------------------------------------
    # 3. Build models
    # ------------------------------------------------------------------
    print("\nBuilding models...")
    agent = AgentPolicy(config)
    world_model = WebWorldModel(config)
    reward_fn = SelfAssessReward(agent, config)

    # ------------------------------------------------------------------
    # 4. Build trainer and run
    # ------------------------------------------------------------------
    trainer = DynaWebTrainer(
        config=config,
        agent=agent,
        world_model=world_model,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ------------------------------------------------------------------
    # 5. Optional evaluation
    # ------------------------------------------------------------------
    if args.eval:
        print("\nRunning evaluation...")
        metrics = trainer.evaluate()
        print(f"Evaluation results: {metrics}")

    print("\nDone.")


if __name__ == "__main__":
    main()