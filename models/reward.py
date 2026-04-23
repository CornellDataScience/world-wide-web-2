"""
models/reward.py

Self-assessment reward: prompts the agent LLM to judge whether
a completed trajectory achieved the task goal.

Returns 1.0 (success) or 0.0 (failure) per trajectory.
Reuses the AgentPolicy model — no separate model needed.
"""

import torch
from typing import List

from config import DynaWebConfig
from data.datatypes import WebInteraction
from utils.prompt import format_reward_prompt


class SelfAssessReward:
    """
    Computes binary task completion reward by prompting the agent LLM.

    Usage:
        reward_fn = SelfAssessReward(agent, config)
        reward = reward_fn.compute_reward(interaction)
        rewards = reward_fn.batch_compute_rewards([i1, i2, i3])
    """

    def __init__(self, agent, config: DynaWebConfig):
        """
        Input:
            agent:  AgentPolicy — the agent model (reused for scoring)
            config: DynaWebConfig
        """
        self.agent = agent
        self.config = config

    def compute_reward(self, interaction: WebInteraction) -> float:
        """
        Compute reward for a single WebInteraction.

        Input:  interaction: WebInteraction
        Output: float — 1.0 or 0.0
        """
        # Real expert trajectories: trust their stored reward
        if interaction.is_real:
            return interaction.reward

        # Imagined trajectories: use self-assessment
        prompt = format_reward_prompt(
            task=interaction.task,
            obs_list=interaction.obs_list(),
            action_list=interaction.action_list(),
        )
        return self.agent.score_reward_prompt(prompt)

    def batch_compute_rewards(
        self,
        interactions: List[WebInteraction],
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of interactions.

        Input:  interactions: List[WebInteraction]  length B
        Output: Tensor[B]  — float rewards
        """
        rewards = []
        for interaction in interactions:
            r = self.compute_reward(interaction)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)