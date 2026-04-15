"""
Entropy-adaptive segmentation.

Given a flat byte sequence (a message), determines word boundaries
by monitoring the per-byte entropy of the model's next-byte prediction.

Key idea: when the model is "surprised" (high entropy), it's still
building up context. When entropy drops sharply, the model is confident
about what comes next — likely because a recognizable unit (word) just
completed. We declare a boundary at local entropy minima.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def byte_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy from logits.

    Args:
        logits: [..., vocab_size]
    Returns:
        entropy: [...]   in nats
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def smooth(signal: torch.Tensor, window: int) -> torch.Tensor:
    """1D box-car smoothing via convolution."""
    if window <= 1 or signal.numel() < window:
        return signal
    kernel = signal.new_ones(window) / window
    # F.conv1d expects [B, C, L]
    s = signal.unsqueeze(0).unsqueeze(0)
    k = kernel.unsqueeze(0).unsqueeze(0)
    pad = window // 2
    smoothed = F.conv1d(s, k, padding=pad).squeeze()
    return smoothed[: signal.numel()]  # trim any extra from padding


def find_boundaries_from_entropy(
    entropies: torch.Tensor,
    threshold: float,
    min_len: int,
    max_len: int,
    smoothing_window: int = 3,
) -> list[int]:
    """
    Identify word boundary positions from a 1-D entropy signal.

    A boundary is placed *after* position i when:
      1. entropy[i] < threshold  (model is confident here)
      2. entropy[i] is a local minimum relative to its neighbours
      3. At least min_len bytes have elapsed since the last boundary
      4. max_len bytes forces a boundary regardless

    Args:
        entropies : [L]  — per-byte entropy values
        threshold : float
        min_len   : int
        max_len   : int
        smoothing_window: int
    Returns:
        boundaries: list of ints — byte positions where a new word starts
                    (position 0 is never listed; it's always a word start)
    """
    signal = smooth(entropies.float(), smoothing_window)
    L = signal.numel()
    boundaries: list[int] = []
    last_boundary = 0

    for i in range(1, L):
        since_last = i - last_boundary

        # Force boundary if we've hit max_len
        if since_last >= max_len:
            boundaries.append(i)
            last_boundary = i
            continue

        if since_last < min_len:
            continue

        # Local minimum check
        prev_e = signal[i - 1].item()
        curr_e = signal[i].item()
        next_e = signal[i + 1].item() if i + 1 < L else float("inf")

        is_local_min = (curr_e <= prev_e) and (curr_e <= next_e)
        if is_local_min and curr_e < threshold:
            boundaries.append(i)
            last_boundary = i

    return boundaries


def segment_bytes(
    byte_ids: list[int],
    entropies: torch.Tensor,
    threshold: float,
    min_len: int = 1,
    max_len: int = 16,
    smoothing_window: int = 3,
) -> list[list[int]]:
    """
    Split a flat byte sequence into word segments using entropy boundaries.

    Args:
        byte_ids  : list of int byte values
        entropies : [len(byte_ids)] entropy at each position
        threshold, min_len, max_len, smoothing_window: segmentation params
    Returns:
        segments: list of byte lists, one per word
    """
    boundaries = find_boundaries_from_entropy(
        entropies, threshold, min_len, max_len, smoothing_window
    )
    # Insert 0 as implicit first boundary, append L as end
    split_points = [0] + boundaries + [len(byte_ids)]
    return [byte_ids[split_points[i]: split_points[i + 1]] for i in range(len(split_points) - 1)]


class EntropySegmenter:
    """
    Wraps a CharacterDecoder (or any model with a generate-logits method)
    to segment an incoming byte stream on-the-fly during inference.

    Usage:
        segmenter = EntropySegmenter(decoder, cfg)
        segments = segmenter.segment(byte_sequence, context_embedding)
    """

    def __init__(self, threshold: float, min_len: int, max_len: int, smoothing_window: int):
        self.threshold = threshold
        self.min_len = min_len
        self.max_len = max_len
        self.smoothing_window = smoothing_window

    def segment_with_logits(
        self, byte_ids: list[int], logits: torch.Tensor
    ) -> list[list[int]]:
        """
        Args:
            byte_ids: flat byte sequence
            logits  : [L, vocab_size] — model logits at each byte position
        Returns:
            segments: list of byte lists
        """
        entropies = byte_entropy(logits)
        return segment_bytes(
            byte_ids,
            entropies,
            threshold=self.threshold,
            min_len=self.min_len,
            max_len=self.max_len,
            smoothing_window=self.smoothing_window,
        )
