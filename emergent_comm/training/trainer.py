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
)


def _prepend_w_token(byte_ids: torch.Tensor) -> torch.Tensor:
    """Prepend W_TOKEN to each sequence in a batch [B, L] → [B, L+1]."""
    B = byte_ids.size(0)
    prefix = byte_ids.new_full((B, 1), W_TOKEN)
    return torch.cat([prefix, byte_ids], dim=1)


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

        self.optimizer = torch.optim.AdamW(
            list(sender.parameters()) + list(receiver.parameters()),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Initialise baseline at random-chance level so REINFORCE has a neutral start
        self.baseline = 1.0 / (cfg.env.n_distractors + 1)
        self.history = TrainingHistory()

    def _step(self, batch: dict) -> dict:
        """One training step. Returns dict of scalar metrics."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        target_attrs = batch["target_attrs"]       # [B, 3]
        candidate_attrs = batch["candidate_attrs"] # [B, N, 3]
        target_idx = batch["target_idx"]           # [B]

        # --- Sender: generate message bytes ---
        sender_out = self.sender(target_attrs)
        msg_bytes = sender_out["message_bytes"]    # [B, gen_len]
        sender_log_probs = sender_out["log_probs"] # [B, gen_len]

        # Prepend W_TOKEN so receiver encoder sees standard format.
        # Fixed-length messages have no padding.
        msg_with_w = _prepend_w_token(msg_bytes)   # [B, 1+msg_len]

        # --- Receiver: score candidates ---
        receiver_out = self.receiver(msg_with_w, candidate_attrs, pad_mask=None)
        recv_log_probs = receiver_out["log_probs"] # [B, N]

        # --- Losses ---
        s_loss, r_loss, self.baseline = reinforce_loss(
            recv_log_probs, target_idx, sender_log_probs, self.baseline
        )

        # Entropy regularisation: penalise low-entropy (repetitive) sender distributions.
        # We use the logits to compute the true Shannon entropy of the policy.
        # Higher entropy = more diverse messages = avoids mode collapse.
        e_loss = sender_entropy_loss(sender_out["logits"])

        total_loss = s_loss + r_loss + self.cfg.training.entropy_coeff * e_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.sender.parameters()) + list(self.receiver.parameters()),
            self.cfg.training.grad_clip,
        )
        self.optimizer.step()

        acc = communication_accuracy(recv_log_probs, target_idx)
        # With fixed-length messages, report unique messages in batch as diversity proxy
        mean_msg_len = msg_bytes.size(1)  # always = max_message_len

        return {
            "acc": acc,
            "sender_loss": s_loss.item(),
            "receiver_loss": r_loss.item(),
            "mean_msg_len": mean_msg_len,
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

            metrics = self._step(batch)
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
                    f"s_loss={metrics['sender_loss']:.4f}  r_loss={metrics['receiver_loss']:.4f}  "
                    f"({elapsed:.0f}s)"
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
