"""
data/dataset.py

NNetNavDataset: loads WebInteraction objects from JSONL files,
or falls back to hardcoded samples if no file is present.

Data directory layout:
    data/raw/train.jsonl    ← put your NNetNav training data here
    data/raw/val.jsonl      ← put your NNetNav validation data here

JSONL format — each line is one of these two formats:

Format A (preferred, explicit pre/post observations):
{
  "task": "Search for Python tutorials",
  "website": "example.com",
  "actions": [
    {
      "action": "type [3] [Python tutorials] [1]",
      "pre_obs": "[textbox id=3 'Search']...",
      "post_obs": "[heading id=10 'Results']...",
      "url": "https://example.com"
    }
  ],
  "reward": 1.0
}

Format B (simpler, obs only at each step):
{
  "task": "...",
  "trajectory": [
    {"obs": "...", "action": "click [7]"},
    {"obs": "...", "action": "stop [done]"}
  ],
  "reward": 1.0
}
"""

import json
import os
import random
from typing import List, Optional, Tuple

try:
    from torch.utils.data import Dataset as TorchDataset
    _BASE = TorchDataset
except ImportError:
    _BASE = object   # graceful fallback when torch is not installed

from data.types import WebInteraction, WebState
from data.samples import make_sample_interactions


class NNetNavDataset(_BASE):
    """
    Dataset of WebInteraction objects.

    Each item is a complete WebInteraction with:
      - task: str
      - state_changes: List[StateChange]  (each has pre_state, action, post_state)
      - reward: float
      - final_state: WebState
    """

    def __init__(self, jsonl_path: str):
        self.interactions: List[WebInteraction] = []
        self._load(jsonl_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            print(f"[Dataset] {path} not found — using {len(make_sample_interactions())} hardcoded samples.")
            self.interactions = make_sample_interactions()
            return

        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    reward = float(raw.get("reward", 1.0))
                    interaction = WebInteraction.from_raw_data(raw, reward=reward)
                    if interaction.num_steps() > 0:
                        self.interactions.append(interaction)
                except Exception as e:
                    print(f"[Dataset] Skipping line {i}: {e}")

        print(f"[Dataset] Loaded {len(self.interactions)} interactions from {path}")

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> WebInteraction:
        return self.interactions[idx]

    def sample_random_state(self) -> Tuple[str, WebState]:
        """
        Return (task, random_state_from_any_trajectory).
        Used by rollout engine to start dreaming from mid-trajectory states,
        per Section 4.1: 'initial states randomly sampled from trajectory data
        including both initial and intermediate states'.
        """
        interaction = random.choice(self.interactions)
        # Pick any state in the trajectory (initial or intermediate)
        all_states = [sc.pre_state for sc in interaction.state_changes]
        state = random.choice(all_states)
        return interaction.task, state

    def get_wm_training_pairs(self) -> List[dict]:
        """
        Flatten all StateChanges across all interactions into (pre, action, post) dicts.
        Used for world model supervised fine-tuning.
        Returns List of dicts with keys: pre_obs, action, post_obs, dom_changed
        """
        pairs = []
        for interaction in self.interactions:
            for sc in interaction.state_changes:
                pairs.append(sc.to_wm_training_dict())
        return pairs


def collate_interactions(batch: List[WebInteraction]) -> List[WebInteraction]:
    """
    Collate function for DataLoader.
    Returns the batch as-is (list of WebInteraction).
    Tensor conversion happens downstream in mixer.py.
    """
    return batch