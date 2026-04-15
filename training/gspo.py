"""
training/gspo.py

Group Sequence Policy Optimization (GSPO) loss.

Implements equations (8), (9), (10) from the paper.

Key insight vs standard PPO:
  - Standard PPO clips token-level ratios r_k = pi_new(y_k) / pi_old(y_k)
  - GSPO lifts importance sampling to the sequence level:
    s_i = geometric_mean(r_k over all response tokens)
  - This single sequence ratio is then clipped and multiplied by advantage
  - Result: more stable training for long multi-step trajectories
"""

import torch
from typing import Optional


def compute_token_ratios(
    new_log_probs: torch.Tensor,   # [B, L]
    old_log_probs: torch.Tensor,   # [B, L]
) -> torch.Tensor:
    """
    Compute per-token likelihood ratios r_k = pi_new / pi_old.
    Works in log space for numerical stability.

    Input:
        new_log_probs: Tensor[B, L]  — log probs under current policy
        old_log_probs: Tensor[B, L]  — log probs under old policy (detached)
    Output:
        Tensor[B, L]  — per-token ratios (in log space: log_new - log_old)
    """
    # Keep in log space: log(r_k) = log_new - log_old
    return new_log_probs - old_log_probs   # [B, L]


def compute_sequence_ratios(
    log_token_ratios: torch.Tensor,   # [B, L]   log(r_k) per token
    seq_lengths: torch.Tensor,        # [B]       number of response tokens
    labels: torch.Tensor,             # [B, L]   -100 for prompt positions
) -> torch.Tensor:
    """
    Compute sequence-level ratio s_i = geometric mean of per-token ratios.
    Equation (9): s_i = exp( (1/|y_i|) * sum_k log(r_k) )

    Only averages over response tokens (where labels != -100).

    Input:
        log_token_ratios: Tensor[B, L]
        seq_lengths:      Tensor[B]
        labels:           Tensor[B, L]
    Output:
        Tensor[B]  — one sequence-level ratio per trajectory
    """
    # Mask: only score response tokens
    response_mask = (labels != -100).float()   # [B, L]

    # Sum log ratios over response tokens
    sum_log_ratios = (log_token_ratios * response_mask).sum(dim=1)   # [B]

    # Divide by sequence length (geometric mean in log space)
    safe_lengths = seq_lengths.float().clamp(min=1.0)
    mean_log_ratio = sum_log_ratios / safe_lengths   # [B]

    # Exponentiate to get sequence ratio
    seq_ratios = torch.exp(mean_log_ratio)   # [B]
    return seq_ratios


def compute_gspo_advantages(
    rewards: torch.Tensor,   # [B*n]
    n: int,                  # rollouts per task
) -> torch.Tensor:
    """
    Compute GRPO-style advantages: normalize within each group of n rollouts.
    Equation (8): advantage = (r - mean_group) / (std_group + eps)

    Input:
        rewards: Tensor[B*n]  — scalar reward per trajectory
        n:       int          — group size (rollouts per task)
    Output:
        Tensor[B*n]  — normalized advantages
    """
    B = rewards.shape[0] // n
    rewards_grouped = rewards.view(B, n)   # [B, n]

    mean = rewards_grouped.mean(dim=1, keepdim=True)   # [B, 1]
    std  = rewards_grouped.std(dim=1, keepdim=True)    # [B, 1]

    advantages = (rewards_grouped - mean) / (std + 1e-8)   # [B, n]
    return advantages.view(-1)                              # [B*n]


def gspo_loss(
    seq_ratios: torch.Tensor,    # [B]
    advantages: torch.Tensor,   # [B]
    eps: float = 0.2,
) -> torch.Tensor:
    """
    Clipped GSPO surrogate objective.
    Equation (8): min( s * A, clip(s, 1-eps, 1+eps) * A )
    Negated for gradient descent (we maximize expected return).

    Input:
        seq_ratios: Tensor[B]  — sequence-level importance ratios
        advantages: Tensor[B]  — normalized advantages
        eps:        float      — clipping threshold (paper: 0.2)
    Output:
        Tensor scalar  — loss value (negated objective)
    """
    clipped = torch.clamp(seq_ratios, 1.0 - eps, 1.0 + eps)

    surrogate_1 = seq_ratios * advantages
    surrogate_2 = clipped * advantages

    # Take element-wise min, then mean over batch
    loss = -torch.min(surrogate_1, surrogate_2).mean()
    return loss


def compute_full_gspo_loss(
    agent,
    old_log_probs: torch.Tensor,   # [B, L]  detached
    input_ids: torch.Tensor,       # [B, L]
    labels: torch.Tensor,          # [B, L]
    rewards: torch.Tensor,         # [B]     (= B*n flattened)
    seq_lengths: torch.Tensor,     # [B]
    n: int,
    eps: float = 0.2,
) -> tuple:
    """
    Full GSPO loss computation: ratios → advantages → clipped loss.

    Input:
        agent:          AgentPolicy
        old_log_probs:  Tensor[B, L]  — from pi_old (detached, no grad)
        input_ids:      Tensor[B, L]
        labels:         Tensor[B, L]
        rewards:        Tensor[B]
        seq_lengths:    Tensor[B]
        n:              int  — rollouts per task
        eps:            float
    Output:
        (loss: Tensor scalar, diagnostics: dict)
    """
    # New log probs under current policy (gradient flows through these)
    new_log_probs = agent.get_token_log_probs(input_ids, labels)   # [B, L]

    # Per-token ratios (log space)
    log_ratios = compute_token_ratios(new_log_probs, old_log_probs)   # [B, L]

    # Sequence-level ratios
    seq_ratios = compute_sequence_ratios(log_ratios, seq_lengths, labels)   # [B]

    # Advantages
    advantages = compute_gspo_advantages(rewards, n)   # [B]

    # Loss
    loss = gspo_loss(seq_ratios, advantages, eps=eps)

    diagnostics = {
        "loss": loss.item(),
        "mean_reward": rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_seq_ratio": seq_ratios.mean().item(),
        "fraction_clipped": (
            (seq_ratios < 1 - eps) | (seq_ratios > 1 + eps)
        ).float().mean().item(),
    }

    return loss, diagnostics