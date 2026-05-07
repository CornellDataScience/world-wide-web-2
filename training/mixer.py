"""
training/mixer.py

TrajectoryMixer: randomly replaces a fraction of imagined trajectories
with real expert trajectories from the NNetNav SFT dataset.

Per Section 3.3: "real expert trajectories are randomly interleaved
with imagined rollouts during training" at ratio real_traj_ratio (50%).

Also handles serialization of WebInteraction objects into tensors
for GSPO loss computation.
"""

import random
import torch
from typing import List, Tuple, Optional

from dynaweb_config import DynaWebConfig
from data.datatypes import WebInteraction
from data.dataset import NNetNavDataset
from utils.prompt import format_agent_prompt


class TrajectoryMixer:
    """
    Mixes imagined and real trajectories, then prepares them for GSPO.

    Usage:
        mixer = TrajectoryMixer(sft_dataset, config, tokenizer)
        flat_batch = mixer.mix(imagined_groups)        # List[WebInteraction]
        gspo_dict  = mixer.to_gspo_batch(flat_batch)  # Dict[str, Tensor]
    """

    def __init__(
        self,
        sft_dataset: NNetNavDataset,
        config: DynaWebConfig,
        tokenizer=None,
    ):
        self.sft_dataset = sft_dataset
        self.config = config
        self.tokenizer = tokenizer

    def mix(
        self,
        imagined_groups: List[List[WebInteraction]],
    ) -> List[WebInteraction]:
        """
        Flatten [B, n] imagined groups and replace real_traj_ratio fraction
        with real expert trajectories.

        Input:
            imagined_groups: List[List[WebInteraction]]  shape [B, n]
        Output:
            List[WebInteraction]  length B*n  (mix of real and imagined)
        """
        flat = [traj for group in imagined_groups for traj in group]

        n_real = int(len(flat) * self.config.real_traj_ratio)
        real_indices = random.sample(range(len(flat)), k=min(n_real, len(flat)))
        real_samples = self._sample_real(len(real_indices))

        for i, idx in enumerate(real_indices):
            flat[idx] = real_samples[i]

        n_imagined = sum(1 for t in flat if not t.is_real)
        n_real_final = sum(1 for t in flat if t.is_real)
        print(f"[Mixer] Batch: {len(flat)} total | {n_imagined} imagined | {n_real_final} real")

        return flat

    def _sample_real(self, n: int) -> List[WebInteraction]:
        """Sample n real trajectories from SFT dataset (with replacement)."""
        indices = [random.randint(0, len(self.sft_dataset) - 1) for _ in range(n)]
        return [self.sft_dataset[i] for i in indices]

    def to_gspo_batch(
        self,
        mixed: List[WebInteraction],
        device: str = "cpu",
    ) -> dict:
        """
        Serialize a list of WebInteractions into tensors for GSPO.

        Each WebInteraction is serialized as a full prompt+response sequence:
          - prompt  = formatted agent prompt (all obs + actions up to current)
          - response = the final action string

        Labels are -100 for prompt tokens (not scored) and token ids for
        response tokens (scored in GSPO ratio).

        Input:
            mixed:  List[WebInteraction]  length B*n
            device: str
        Output:
            {
              "input_ids":      Tensor[B*n, max_L]
              "attention_mask": Tensor[B*n, max_L]
              "labels":         Tensor[B*n, max_L]   (-100 for prompt)
              "rewards":        Tensor[B*n]
              "seq_lengths":    Tensor[B*n]           (response token counts)
              "is_real":        Tensor[B*n]           (bool, 1=real 0=imagined)
            }
        """
        if self.tokenizer is None:
            return self._stub_gspo_batch(mixed, device)

        all_input_ids = []
        all_attention = []
        all_labels = []
        all_rewards = []
        all_seq_lengths = []
        all_is_real = []

        for interaction in mixed:
            obs = interaction.obs_list()
            acts = interaction.action_list()

            if not obs or not acts:
                continue

            # Build prompt from all steps except the last action
            prompt = format_agent_prompt(
                task=interaction.task,
                obs_history=obs[:-1] if len(obs) > 1 else obs,
                action_history=acts[:-1] if len(acts) > 1 else [],
            )
            response = acts[-1] if acts else "stop []"

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)

            full_ids = prompt_ids + response_ids
            prompt_len = len(prompt_ids)
            response_len = len(response_ids)

            # Truncate if needed
            max_len = self.config.max_prompt_len + self.config.max_response_len
            full_ids = full_ids[:max_len]
            prompt_len = min(prompt_len, len(full_ids))

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]

            all_input_ids.append(full_ids)
            all_attention.append([1] * len(full_ids))
            all_labels.append(labels)
            all_rewards.append(interaction.reward)
            all_seq_lengths.append(max(1, len(full_ids) - prompt_len))
            all_is_real.append(1 if interaction.is_real else 0)

        if not all_input_ids:
            return self._stub_gspo_batch(mixed, device)

        # Pad all sequences to same length
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_input_ids]
        padded_att = [att + [0] * (max_len - len(att)) for att in all_attention]
        padded_lab = [lab + [-100] * (max_len - len(lab)) for lab in all_labels]

        return {
            "input_ids":      torch.tensor(padded_ids, dtype=torch.long).to(device),
            "attention_mask": torch.tensor(padded_att, dtype=torch.long).to(device),
            "labels":         torch.tensor(padded_lab, dtype=torch.long).to(device),
            "rewards":        torch.tensor(all_rewards, dtype=torch.float32).to(device),
            "seq_lengths":    torch.tensor(all_seq_lengths, dtype=torch.long).to(device),
            "is_real":        torch.tensor(all_is_real, dtype=torch.bool).to(device),
        }

    def _stub_gspo_batch(self, mixed: List[WebInteraction], device: str) -> dict:
        """
        Stub batch for testing without a tokenizer.
        Creates plausible-shaped tensors with random values.
        """
        B = len(mixed)
        L = 32  # fake sequence length

        return {
            "input_ids":      torch.randint(0, 1000, (B, L)).to(device),
            "attention_mask": torch.ones(B, L, dtype=torch.long).to(device),
            "labels":         torch.cat([
                torch.full((B, L // 2), -100, dtype=torch.long),
                torch.randint(0, 1000, (B, L // 2))
            ], dim=1).to(device),
            "rewards":        torch.tensor(
                [t.reward for t in mixed], dtype=torch.float32
            ).to(device),
            "seq_lengths":    torch.full((B,), L // 2, dtype=torch.long).to(device),
            "is_real":        torch.tensor(
                [t.is_real for t in mixed], dtype=torch.bool
            ).to(device),
        }