"""
The signaling game world: generates batches of (target, candidates) trials
and converts object attributes into tensors for the model.
"""

from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from .entities import Object, SignalingTrial, all_objects, make_trial
from ..config import EnvConfig


class SignalingGameDataset(IterableDataset):
    """
    Infinite stream of signaling game trials.

    Each item is a dict:
        target_attrs   : LongTensor [3]          — (color, shape, size)
        candidate_attrs: LongTensor [n_cands, 3] — all candidates' attributes
        target_idx     : LongTensor []            — index of target in candidates
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.object_pool = all_objects(cfg.n_colors, cfg.n_shapes, cfg.n_sizes)
        self.rng = random.Random(cfg.seed)

    def __iter__(self) -> Iterator[dict]:
        while True:
            target = self.rng.choice(self.object_pool)
            trial = make_trial(
                target=target,
                object_pool=self.object_pool,
                n_distractors=self.cfg.n_distractors,
                rng=self.rng,
            )
            yield self._trial_to_tensors(trial)

    def _trial_to_tensors(self, trial: SignalingTrial) -> dict:
        target_attrs = torch.tensor(trial.target.to_vector(), dtype=torch.long)
        candidate_attrs = torch.tensor(
            [c.to_vector() for c in trial.candidates], dtype=torch.long
        )
        target_idx = torch.tensor(trial.target_idx, dtype=torch.long)
        return {
            "target_attrs": target_attrs,
            "candidate_attrs": candidate_attrs,
            "target_idx": target_idx,
        }

    @property
    def n_candidates(self) -> int:
        return self.cfg.n_distractors + 1

    @property
    def attribute_sizes(self) -> tuple[int, int, int]:
        return (self.cfg.n_colors, self.cfg.n_shapes, self.cfg.n_sizes)


def collate_trials(batch: list[dict]) -> dict:
    """Stack a list of trial dicts into batched tensors."""
    return {
        "target_attrs": torch.stack([b["target_attrs"] for b in batch]),
        "candidate_attrs": torch.stack([b["candidate_attrs"] for b in batch]),
        "target_idx": torch.stack([b["target_idx"] for b in batch]),
    }
