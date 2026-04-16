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

    Color is weighted 2× because it is empirically the weakest attribute
    (disentanglement ~0.07 vs ~0.25 for shape/size) and has the most classes.

    Args:
        color_logits : [B, n_colors]
        shape_logits : [B, n_shapes]
        size_logits  : [B, n_sizes]
        target_attrs : [B, 3]  — (color_idx, shape_idx, size_idx)
    Returns:
        weighted mean cross-entropy over the three attributes
    """
    color_loss = F.cross_entropy(color_logits, target_attrs[:, 0])
    shape_loss = F.cross_entropy(shape_logits, target_attrs[:, 1])
    size_loss  = F.cross_entropy(size_logits,  target_attrs[:, 2])
    # Color gets 2× weight: 8 values but worst disentanglement
    return (2.0 * color_loss + shape_loss + size_loss) / 4.0


def soft_topsim_loss(
    soft_tokens: torch.Tensor,
    target_attrs: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable approximation of topographic similarity (topsim).

    Topsim = Spearman correlation between pairwise object distances and
    pairwise message distances. We approximate it by MSE-matching normalized
    pairwise distances in a batch, providing a direct gradient for making
    similar objects produce similar messages.

    Args:
        soft_tokens  : [B, L, V]  — Gumbel-softmax one-hots from sender
        target_attrs : [B, 3]     — (color_idx, shape_idx, size_idx)
    Returns:
        scalar loss (lower = messages and objects more correlated)
    """
    B, L, V = soft_tokens.shape

    # Pairwise L2 distances between flattened soft message vectors [B, L*V]
    msg_flat = soft_tokens.view(B, -1)
    msg_dist = torch.cdist(msg_flat, msg_flat, p=2)  # [B, B]

    # Pairwise Hamming distances between attribute tuples [B, B]
    obj_dist = (target_attrs.unsqueeze(0) != target_attrs.unsqueeze(1)).float().mean(dim=-1)

    # Use upper triangle only (avoid diagonal and double-counting)
    mask = torch.triu(torch.ones(B, B, device=soft_tokens.device, dtype=torch.bool), diagonal=1)
    msg_d = msg_dist[mask]
    obj_d = obj_dist[mask]

    # Normalize both to zero mean, unit std before comparing
    msg_d = (msg_d - msg_d.mean()) / (msg_d.std() + 1e-8)
    obj_d = (obj_d - obj_d.mean()) / (obj_d.std() + 1e-8)

    # MSE: pushes message distance structure to match object distance structure
    return F.mse_loss(msg_d, obj_d)


def communication_accuracy(receiver_log_probs: torch.Tensor, target_idx: torch.Tensor) -> float:
    """Fraction of trials where receiver correctly identified the target."""
    predicted = receiver_log_probs.argmax(dim=-1)
    return (predicted == target_idx).float().mean().item()
