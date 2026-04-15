"""
Defines the objects that exist in the signaling game world.

Each object is a named tuple of discrete attributes. The sender
observes one target object and must communicate it to the receiver,
who selects from a candidate pool containing the target + distractors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


# Human-readable attribute names used for logging / compositionality analysis
COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
SHAPE_NAMES = ["circle", "square", "triangle", "star", "cross", "diamond", "hexagon", "arrow"]
SIZE_NAMES = ["tiny", "small", "medium", "large"]


@dataclass(frozen=True)
class Object:
    """A single object in the world, described by discrete attributes."""
    color: int   # index into COLOR_NAMES
    shape: int   # index into SHAPE_NAMES
    size: int    # index into SIZE_NAMES

    def to_vector(self) -> tuple[int, int, int]:
        return (self.color, self.shape, self.size)

    def to_label(self) -> str:
        return f"{SIZE_NAMES[self.size]} {COLOR_NAMES[self.color]} {SHAPE_NAMES[self.shape]}"

    def __repr__(self) -> str:
        return f"Object({self.to_label()})"


def all_objects(n_colors: int, n_shapes: int, n_sizes: int) -> list[Object]:
    """Return every possible object given the attribute counts."""
    return [
        Object(color=c, shape=s, size=z)
        for c in range(n_colors)
        for s in range(n_shapes)
        for z in range(n_sizes)
    ]


@dataclass
class SignalingTrial:
    """
    One trial of the communication game.

    The sender observes `target`. The receiver sees `candidates`
    (which contains `target` at index `target_idx`) and must
    identify which one the sender is referring to.
    """
    target: Object
    candidates: list[Object]
    target_idx: int

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def make_trial(
    target: Object,
    object_pool: Sequence[Object],
    n_distractors: int,
    rng: random.Random,
) -> SignalingTrial:
    """
    Build a trial: pick `n_distractors` objects from `object_pool`
    (excluding `target`), shuffle with the target, record target index.
    """
    distractors = rng.sample(
        [o for o in object_pool if o != target],
        k=min(n_distractors, len(object_pool) - 1),
    )
    candidates = distractors + [target]
    rng.shuffle(candidates)
    target_idx = candidates.index(target)
    return SignalingTrial(target=target, candidates=candidates, target_idx=target_idx)
