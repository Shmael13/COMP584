"""
Loss functions for the emergent communication game.

Two training modes:
  1. Differentiable (straight-through / Gumbel-softmax):
     Sender produces soft byte distributions; receiver gets soft embeddings.
     End-to-end differentiable but approximate.

  2. REINFORCE:
     Sender samples discrete bytes; receiver gets hard byte sequence.
     Exact but high-variance gradient estimator.
     We use a running mean baseline to reduce variance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def receiver_nll_loss(log_probs: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood loss for the receiver's prediction.

    Args:
        log_probs : [B, n_cands]
        target_idx: [B]
    Returns:
        scalar loss
    """
    return F.nll_loss(log_probs, target_idx)


def sender_entropy_loss(logits: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Entropy regularization on the sender's byte distribution.
    Maximizing entropy encourages diverse messages (avoids mode collapse).

    Args:
        logits  : [B, msg_len, vocab_size]
        pad_mask: [B, msg_len] — True where padded (excluded from entropy)
    Returns:
        negative mean entropy (to be *subtracted* from total loss)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, msg_len]

    if pad_mask is not None:
        active = ~pad_mask
        entropy = (entropy * active).sum() / active.float().sum().clamp(min=1)
    else:
        entropy = entropy.mean()

    return -entropy  # negate: minimizing this maximizes entropy


def reinforce_loss(
    receiver_log_probs: torch.Tensor,
    target_idx: torch.Tensor,
    sender_log_probs: torch.Tensor,
    baseline: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    REINFORCE loss for the sender.

    Reward = 1 if receiver chose correctly, 0 otherwise.
    Baseline-subtracted to reduce variance.

    Args:
        receiver_log_probs: [B, n_cands]
        target_idx        : [B]
        sender_log_probs  : [B, msg_len] — log-prob of each sampled byte
        baseline          : running mean reward (float)
    Returns:
        sender_loss : scalar
        receiver_loss: scalar
        new_baseline: updated running mean
    """
    # Compute per-sample reward (0 or 1)
    predicted = receiver_log_probs.argmax(dim=-1)               # [B]
    reward = (predicted == target_idx).float()                  # [B]

    # Update running baseline (faster decay = quicker adaptation)
    mean_reward = reward.mean().item()
    new_baseline = 0.95 * baseline + 0.05 * mean_reward

    # Advantage
    advantage = (reward - baseline).detach()                    # [B]

    # Sender loss: -advantage * sum of log-probs over message bytes
    total_log_prob = sender_log_probs.sum(dim=-1)               # [B]
    sender_loss = -(advantage * total_log_prob).mean()

    # Receiver loss: standard cross-entropy
    receiver_loss = receiver_nll_loss(receiver_log_probs, target_idx)

    return sender_loss, receiver_loss, new_baseline


def attribute_prediction_loss(
    color_logits: torch.Tensor,
    shape_logits: torch.Tensor,
    size_logits: torch.Tensor,
    target_attrs: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary compositionality loss.

    Forces the receiver's message embedding to predict each object attribute
    (color, shape, size) independently. Because gradients flow back through
    the soft tokens to the sender, this directly pressures the sender to encode
    each attribute as a consistent, separable part of the message rather than
    emitting one arbitrary holistic signal per object.

    Args:
        color_logits : [B, n_colors]
        shape_logits : [B, n_shapes]
        size_logits  : [B, n_sizes]
        target_attrs : [B, 3]  — (color_idx, shape_idx, size_idx)
    Returns:
        mean cross-entropy over the three attributes
    """
    return (
        F.cross_entropy(color_logits, target_attrs[:, 0]) +
        F.cross_entropy(shape_logits, target_attrs[:, 1]) +
        F.cross_entropy(size_logits, target_attrs[:, 2])
    ) / 3.0


def communication_accuracy(receiver_log_probs: torch.Tensor, target_idx: torch.Tensor) -> float:
    """Fraction of trials where receiver correctly identified the target."""
    predicted = receiver_log_probs.argmax(dim=-1)
    return (predicted == target_idx).float().mean().item()
