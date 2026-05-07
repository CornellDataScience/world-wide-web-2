"""
training/trainer.py

DynaWebTrainer: the main training loop.

Per-step order:
  1. Sample initial states from dataset
  2. Generate imagined rollouts (rollout engine)
  3. Mix with real expert trajectories (mixer)
  4. Compute old log probs (no grad)
  5. Serialize to tensors (mixer.to_gspo_batch)
  6. Run GSPO loss (new log probs, ratio, clip, advantage)
  7. Backprop + optimizer step
"""

import os
import json
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import List, Dict, Optional

from dynaweb_config import DynaWebConfig
from data.datatypes import WebInteraction, WebState
from data.dataset import NNetNavDataset, collate_interactions
from models.agent import AgentPolicy
from models.world_model import WebWorldModel
from models.reward import SelfAssessReward
from training.rollout import DreamingRolloutEngine
from training.mixer import TrajectoryMixer
from training.gspo import compute_full_gspo_loss


class DynaWebTrainer:

    def __init__(
        self,
        config: DynaWebConfig,
        agent: AgentPolicy,
        world_model: WebWorldModel,
        reward_fn: SelfAssessReward,
        train_dataset: NNetNavDataset,
        val_dataset: Optional[NNetNavDataset] = None,
        tokenizer=None,
    ):
        self.config = config
        self.agent = agent
        self.world_model = world_model
        self.reward_fn = reward_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        self.rollout_engine = DreamingRolloutEngine(agent, world_model, reward_fn, config)
        self.mixer = TrajectoryMixer(train_dataset, config, tokenizer)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, agent.parameters()),
            lr=config.lr,
        ) if agent.model is not None else None

        self.step_count = 0
        self.log_history: List[Dict] = []

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train(self):
        """Outer training loop over epochs and batches."""
        print(f"\n{'='*60}")
        print(f"DynaWeb Training")
        print(f"  Epochs:       {self.config.epochs}")
        print(f"  Batch size:   {self.config.train_batch_size}")
        print(f"  Rollouts/task:{self.config.rollout_n}")
        print(f"  Dream length: {self.config.dream_length}")
        print(f"  Real ratio:   {self.config.real_traj_ratio}")
        print(f"  Stub mode:    {self.config.use_stub_models}")
        print(f"{'='*60}\n")

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collate_interactions,
        )

        for epoch in range(self.config.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.epochs} ---")
            epoch_losses = []

            for batch_idx, batch in enumerate(loader):
                # batch is List[WebInteraction] of length train_batch_size
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["loss"])
                self.step_count += 1

                print(
                    f"  Step {self.step_count:04d} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"reward={metrics['mean_reward']:.3f} | "
                    f"adv={metrics['mean_advantage']:.3f} | "
                    f"ratio={metrics['mean_seq_ratio']:.4f}"
                )

                self.log_history.append({"step": self.step_count, **metrics})

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            self.save_checkpoint(
                os.path.join(self.config.checkpoint_dir, f"epoch_{epoch+1}.pt")
            )

        # Save training log
        log_path = os.path.join(self.config.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=2)
        print(f"\nTraining log saved to {log_path}")

    def train_step(self, batch: List[WebInteraction]) -> Dict:
        """
        One full training step on a batch of tasks.

        Input:  batch: List[WebInteraction]  — length train_batch_size
        Output: metrics dict with loss, reward, advantage, etc.
        """
        device = self.config.device if not self.config.use_stub_models else "cpu"

        # ------------------------------------------------------------------
        # 1. Extract tasks + initial states from batch
        # ------------------------------------------------------------------
        tasks = [interaction.task for interaction in batch]
        initial_states = [
            interaction.state_changes[0].pre_state
            if interaction.state_changes
            else WebState.from_acc_tree("")
            for interaction in batch
        ]
        websites = [interaction.website for interaction in batch]

        # ------------------------------------------------------------------
        # 2. Generate imagined rollouts: [B, n] groups of WebInteractions
        # ------------------------------------------------------------------
        imagined_groups = self.rollout_engine.generate_batch(
            tasks=tasks,
            initial_states=initial_states,
            websites=websites,
        )

        # ------------------------------------------------------------------
        # 3. Mix with real expert trajectories → flat List[WebInteraction] B*n
        # ------------------------------------------------------------------
        mixed = self.mixer.mix(imagined_groups)

        # ------------------------------------------------------------------
        # 4. Serialize to tensors
        # ------------------------------------------------------------------
        gspo_dict = self.mixer.to_gspo_batch(mixed, device=device)

        # ------------------------------------------------------------------
        # 5. Compute old log probs (detached — these are the reference policy)
        # ------------------------------------------------------------------
        with torch.no_grad():
            old_log_probs = self.agent.get_token_log_probs(
                gspo_dict["input_ids"],
                gspo_dict["labels"],
            ).detach()

        # ------------------------------------------------------------------
        # 6. Compute GSPO loss (new log probs computed inside, with grad)
        # ------------------------------------------------------------------
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        loss, diagnostics = compute_full_gspo_loss(
            agent=self.agent,
            old_log_probs=old_log_probs,
            input_ids=gspo_dict["input_ids"],
            labels=gspo_dict["labels"],
            rewards=gspo_dict["rewards"],
            seq_lengths=gspo_dict["seq_lengths"],
            n=self.config.rollout_n,
            eps=self.config.eps_clip,
        )

        # ------------------------------------------------------------------
        # 7. Backprop (skip if stub or no grad)
        # ------------------------------------------------------------------
        if self.optimizer is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        return diagnostics

    def evaluate(self) -> Dict:
        """
        Simple evaluation: run agent greedily on val dataset,
        check if final action is a successful stop.
        Returns success rate.
        """
        if self.val_dataset is None:
            return {"val_success_rate": 0.0}

        n_success = 0
        n_total = len(self.val_dataset)

        for interaction in self.val_dataset:
            # Generate one trajectory (no randomness in eval — use step 0 action repeatedly)
            traj = self.rollout_engine.generate_single_trajectory(
                task=interaction.task,
                initial_state=interaction.state_changes[0].pre_state
                if interaction.state_changes
                else WebState.from_acc_tree(""),
            )
            if traj.was_successful():
                n_success += 1

        sr = n_success / n_total if n_total > 0 else 0.0
        print(f"[Eval] Success rate: {sr:.3f} ({n_success}/{n_total})")
        return {"val_success_rate": sr}

    def save_checkpoint(self, path: str):
        """Save agent weights and optimizer state."""
        checkpoint = {
            "step": self.step_count,
            "config": self.config.__dict__,
            "log_history": self.log_history,
        }
        if self.agent.model is not None:
            checkpoint["agent_state_dict"] = self.agent.state_dict()
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path: str):
        """Load agent weights and optimizer state."""
        checkpoint = torch.load(path, map_location="cpu")
        if "agent_state_dict" in checkpoint and self.agent.model is not None:
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step", 0)
        self.log_history = checkpoint.get("log_history", [])
        print(f"[Checkpoint] Loaded from {path} (step {self.step_count})")