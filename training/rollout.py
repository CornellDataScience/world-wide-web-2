"""
training/rollout.py

DreamingRolloutEngine: generates imagined WebInteraction objects
by having the agent policy interact with the world model.

Per Section 3.3:
  - Start from initial WebState
  - Agent generates action → World model predicts next state
  - Repeat up to dream_length steps
  - Assign reward via self-assessment
  - Return imagined WebInteraction (is_real=False)
"""

import torch
from typing import List, Tuple

from dynaweb_config import DynaWebConfig
from data.types import WebInteraction, WebState, WebAction, StateChange
from models.agent import AgentPolicy
from models.world_model import WebWorldModel
from models.reward import SelfAssessReward
from utils.prompt import format_agent_prompt


class DreamingRolloutEngine:
    """
    Generates imagined trajectories for model-based RL.

    generate_single_trajectory() — one imagined episode
    generate_group()             — n rollouts for one task
    generate_batch()             — n rollouts for each of B tasks
    """

    def __init__(
        self,
        agent: AgentPolicy,
        world_model: WebWorldModel,
        reward_fn: SelfAssessReward,
        config: DynaWebConfig,
    ):
        self.agent = agent
        self.world_model = world_model
        self.reward_fn = reward_fn
        self.config = config

    def generate_single_trajectory(
        self,
        task: str,
        initial_state: WebState,
        website: str = "",
    ) -> WebInteraction:
        """
        Generate one imagined trajectory.

        Input:
            task:          str      — natural language task
            initial_state: WebState — starting page state
            website:       str      — website identifier
        Output:
            WebInteraction with is_real=False, containing all StateChanges
        """
        state_changes: List[StateChange] = []
        obs_history: List[str] = [initial_state.to_model_input()]
        action_history: List[str] = []
        current_state = initial_state
        prev_action: WebAction = None

        for step in range(self.config.dream_length):
            # 1. Agent generates action
            prompt = format_agent_prompt(
                task=task,
                obs_history=obs_history,
                action_history=action_history,
            )
            action_str, token_log_probs = self.agent.generate_action(prompt, step=step)

            # 2. Parse action string into WebAction
            action = WebAction.from_string(action_str)

            # 3. Check terminal
            if self.world_model.is_terminal_state(action):
                # Still record this final step
                state_changes.append(StateChange(
                    pre_state=current_state,
                    action=action,
                    post_state=current_state,
                ))
                # Store log probs on action for GSPO
                action._token_log_probs = token_log_probs
                break

            # 4. World model predicts next state
            state_changes_desc, next_state = self.world_model.predict_next_state(
                task=task,
                current_state=current_state,
                action=action,
                prev_action=prev_action,
            )

            # 5. Store log probs on action object for GSPO later
            action._token_log_probs = token_log_probs

            # 6. Record this step
            sc = StateChange(
                pre_state=current_state,
                action=action,
                post_state=next_state,
            )
            state_changes.append(sc)

            # 7. Advance state
            obs_history.append(next_state.to_model_input())
            action_history.append(action_str)
            prev_action = action
            current_state = next_state

        final_state = state_changes[-1].post_state if state_changes else initial_state

        # Build imagined WebInteraction (reward assigned below)
        interaction = WebInteraction(
            task=task,
            website=website,
            state_changes=state_changes,
            reward=0.0,           # placeholder; filled by reward_fn
            final_state=final_state,
            is_real=False,
        )

        # 8. Assign reward via self-assessment
        interaction.reward = self.reward_fn.compute_reward(interaction)

        return interaction

    def generate_group(
        self,
        task: str,
        initial_state: WebState,
        n: int,
        website: str = "",
    ) -> List[WebInteraction]:
        """
        Generate n imagined trajectories for a single task.
        These form the 'group' used for advantage normalization in GSPO.

        Input:
            task:          str
            initial_state: WebState
            n:             int      — number of rollouts (config.rollout_n)
        Output:
            List[WebInteraction] length n, all with is_real=False
        """
        trajectories = []
        for _ in range(n):
            traj = self.generate_single_trajectory(task, initial_state, website)
            trajectories.append(traj)
        return trajectories

    def generate_batch(
        self,
        tasks: List[str],
        initial_states: List[WebState],
        websites: List[str] = None,
    ) -> List[List[WebInteraction]]:
        """
        Generate n rollouts for each of B tasks.

        Input:
            tasks:          List[str]       length B
            initial_states: List[WebState]  length B
            websites:       List[str]       length B (optional)
        Output:
            List[List[WebInteraction]] shape [B, n]
            Outer index = task, inner index = rollout number
        """
        if websites is None:
            websites = [""] * len(tasks)

        groups = []
        for task, state, website in zip(tasks, initial_states, websites):
            group = self.generate_group(
                task=task,
                initial_state=state,
                n=self.config.rollout_n,
                website=website,
            )
            groups.append(group)
            print(
                f"[Rollout] Task: '{task[:50]}' | "
                f"rewards: {[g.reward for g in group]} | "
                f"steps: {[g.num_steps() for g in group]}"
            )

        return groups