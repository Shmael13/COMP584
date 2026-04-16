"""
Training loop for the emergent communication experiment.

Each step:
  1. Sample a batch of (target, candidates) trials
  2. Sender generates a message (byte stream) from the target object
  3. Receiver reads the message and scores all candidates
  4. Compute loss (REINFORCE for sender, NLL for receiver)
  5. Backprop and update
  6. Periodically evaluate and log metrics
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..agents.sender import Sender
from ..agents.receiver import Receiver
from ..models.encoder import W_TOKEN, PAD_TOKEN
from .objectives import (
    reinforce_loss,
    sender_entropy_loss,
    communication_accuracy,
    receiver_nll_loss,
    attribute_prediction_loss,
)


def _prepend_w_token(byte_ids: torch.Tensor) -> torch.Tensor:
    """Prepend W_TOKEN to each sequence in a batch [B, L] → [B, L+1]."""
    B = byte_ids.size(0)
    prefix = byte_ids.new_full((B, 1), W_TOKEN)
    return torch.cat([prefix, byte_ids], dim=1)


def _prepend_w_token_soft(soft_tokens: torch.Tensor) -> torch.Tensor:
    """Prepend a hard W_TOKEN one-hot to soft token sequence [B, L, V] → [B, L+1, V]."""
    B, _, V = soft_tokens.shape
    w_one_hot = torch.zeros(B, 1, V, device=soft_tokens.device, dtype=soft_tokens.dtype)
    w_one_hot[:, 0, W_TOKEN] = 1.0
    return torch.cat([w_one_hot, soft_tokens], dim=1)


def _make_pad_mask(byte_ids: torch.Tensor, eos_byte: int = 0) -> torch.Tensor:
    """
    Create a padding mask: True for positions *after* the first EOS token.
    byte_ids: [B, L]
    """
    eos_positions = (byte_ids == eos_byte).long()
    # Cumsum > 0 means we've seen at least one EOS
    after_eos = eos_positions.cumsum(dim=1) > 0
    # Shift right: the EOS itself is still valid, everything after is padding
    pad_mask = torch.cat([torch.zeros_like(after_eos[:, :1]), after_eos[:, :-1]], dim=1)
    return pad_mask


@dataclass
class TrainingHistory:
    steps: list[int] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    sender_loss: list[float] = field(default_factory=list)
    receiver_loss: list[float] = field(default_factory=list)
    mean_msg_len: list[float] = field(default_factory=list)


class EmergentCommTrainer:
    def __init__(
        self,
        sender: Sender,
        receiver: Receiver,
        cfg: ExperimentConfig,
        device: torch.device,
    ):
        self.sender = sender.to(device)
        self.receiver = receiver.to(device)
        self.cfg = cfg
        self.device = device

        # Separate optimizers: receiver needs a higher LR so it can track the sender,
        # which changes more slowly (0.5x) to keep messages stable.
        self.sender_opt = torch.optim.AdamW(
            sender.parameters(),
            lr=cfg.training.learning_rate * 0.5,
            weight_decay=cfg.training.weight_decay,
        )
        self.receiver_opt = torch.optim.AdamW(
            receiver.parameters(),
            lr=cfg.training.learning_rate * 2.0,
            weight_decay=cfg.training.weight_decay,
        )

        # Baseline for REINFORCE fallback (not used in straight-through mode)
        self.baseline = 1.0 / (cfg.env.n_distractors + 1)
        self.history = TrainingHistory()

    def _compute_tau(self, step: int) -> float:
        """Gumbel temperature schedule with warmup then exponential decay.

        Phase 1 (warmup): hold at tau_start so the receiver builds stable
          message-meaning associations before messages start shifting.
        Phase 2 (anneal): exponentially decay tau_start → tau_end so messages
          commit to consistent, low-entropy signals that the segmenter can parse
          into reusable word units.
        """
        if not self.cfg.agent.use_straight_through:
            return self.cfg.agent.temperature
        cfg = self.cfg.training
        warmup_steps = int(cfg.n_steps * cfg.gumbel_warmup_frac)
        if step < warmup_steps:
            return cfg.gumbel_tau_start
        progress = (step - warmup_steps) / max(cfg.n_steps - warmup_steps, 1)
        return cfg.gumbel_tau_start * (cfg.gumbel_tau_end / cfg.gumbel_tau_start) ** progress

    def _step(self, batch: dict, step: int) -> dict:
        """One training step. Returns dict of scalar metrics."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        target_attrs = batch["target_attrs"]       # [B, 3]
        candidate_attrs = batch["candidate_attrs"] # [B, N, 3]
        target_idx = batch["target_idx"]           # [B]

        use_st = self.cfg.agent.use_straight_through
        tau = self._compute_tau(step)

        # --- Sender: generate message ---
        sender_out = self.sender(target_attrs, temperature=tau)
        msg_bytes = sender_out["message_bytes"]    # [B, gen_len]
        msg_with_w = _prepend_w_token(msg_bytes)   # [B, 1+gen_len]

        # --- Receiver: score candidates ---
        if use_st and "message_soft" in sender_out:
            # Straight-through: pass differentiable soft tokens so gradients flow
            # from receiver loss all the way back through the sender.
            soft_with_w = _prepend_w_token_soft(sender_out["message_soft"])
            receiver_out = self.receiver(msg_with_w, candidate_attrs, soft_tokens=soft_with_w)
        else:
            receiver_out = self.receiver(msg_with_w, candidate_attrs)

        recv_log_probs = receiver_out["log_probs"]  # [B, N]

        # --- Losses ---
        e_loss = sender_entropy_loss(sender_out["logits"])

        if use_st:
            # End-to-end differentiable: receiver NLL gradient flows through soft tokens
            # back to sender parameters. No REINFORCE needed.
            r_loss = receiver_nll_loss(recv_log_probs, target_idx)
            s_loss = torch.zeros(1, device=self.device)
            total_loss = r_loss + self.cfg.training.entropy_coeff * e_loss
        else:
            s_loss, r_loss, self.baseline = reinforce_loss(
                recv_log_probs, target_idx, sender_out["log_probs"], self.baseline
            )
            total_loss = s_loss + r_loss + self.cfg.training.entropy_coeff * e_loss

        # --- Attribute prediction auxiliary loss ---
        # Forces the message embedding to encode color, shape, and size as
        # separable components. Gradient flows back through soft tokens to the
        # sender, pushing it toward compositional rather than holistic messages.
        msg_emb = receiver_out["msg_embedding"]
        color_l, shape_l, size_l = self.receiver.attribute_logits(msg_emb)
        attr_loss = attribute_prediction_loss(color_l, shape_l, size_l, target_attrs)
        total_loss = total_loss + self.cfg.training.attr_pred_coeff * attr_loss

        self.sender_opt.zero_grad(set_to_none=True)
        self.receiver_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.sender.parameters()) + list(self.receiver.parameters()),
            self.cfg.training.grad_clip,
        )
        self.sender_opt.step()
        self.receiver_opt.step()

        acc = communication_accuracy(recv_log_probs, target_idx)
        return {
            "acc": acc,
            "sender_loss": s_loss.item(),
            "receiver_loss": r_loss.item(),
            "attr_loss": attr_loss.item(),
            "tau": tau,
            "mean_msg_len": msg_bytes.size(1),
        }

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, max_batches: int = 20) -> float:
        self.sender.eval()
        self.receiver.eval()
        total_correct = 0
        total = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            sender_out = self.sender(batch["target_attrs"])
            msg_bytes = sender_out["message_bytes"]
            msg_with_w = _prepend_w_token(msg_bytes)
            receiver_out = self.receiver(msg_with_w, batch["candidate_attrs"], pad_mask=None)
            predicted = receiver_out["predicted_idx"]
            total_correct += (predicted == batch["target_idx"]).sum().item()
            total += batch["target_idx"].size(0)
        self.sender.train()
        self.receiver.train()
        return total_correct / max(total, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        cfg = self.cfg.training
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        best_val_acc = 0.0
        step = 0
        t0 = time.time()

        print(f"Starting training for {cfg.n_steps} steps...")
        for batch in train_loader:
            if step >= cfg.n_steps:
                break

            metrics = self._step(batch, step)
            step += 1

            if step % cfg.eval_every == 0:
                val_acc = self._evaluate(val_loader)
                elapsed = time.time() - t0

                self.history.steps.append(step)
                self.history.train_acc.append(metrics["acc"])
                self.history.val_acc.append(val_acc)
                self.history.sender_loss.append(metrics["sender_loss"])
                self.history.receiver_loss.append(metrics["receiver_loss"])
                self.history.mean_msg_len.append(metrics["mean_msg_len"])

                print(
                    f"step={step:>6}  train_acc={metrics['acc']:.3f}  val_acc={val_acc:.3f}  "
                    f"r_loss={metrics['receiver_loss']:.4f}  attr_loss={metrics['attr_loss']:.4f}  "
                    f"tau={metrics['tau']:.3f}  ({elapsed:.0f}s)"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        {
                            "sender": self.sender.state_dict(),
                            "receiver": self.receiver.state_dict(),
                            "step": step,
                            "val_acc": val_acc,
                        },
                        os.path.join(cfg.checkpoint_dir, "best.pt"),
                    )

        print(f"Training complete. Best val accuracy: {best_val_acc:.3f}")
        return self.history
