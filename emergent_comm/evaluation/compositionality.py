"""
Compositionality metrics for emergent language analysis.

Key metric: Topographic Similarity (topsim)
    Measures the correlation between pairwise distances in meaning space
    (object attributes) and pairwise distances in message space (edit distance).

    High topsim → compositional language: similar objects get similar messages.
    Low topsim → holistic language: no systematic structure.

Additional metrics:
    - Positional disentanglement: does each byte position correlate with one attribute?
    - Vocabulary statistics: unique emergent "words" after entropy segmentation
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Sequence

import torch
import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def attribute_distance(a: tuple, b: tuple) -> int:
    """Hamming distance between two attribute tuples."""
    return sum(x != y for x, y in zip(a, b))


def edit_distance(s1: list[int], s2: list[int]) -> int:
    """Standard Levenshtein edit distance between two byte sequences."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Topographic similarity
# ---------------------------------------------------------------------------

def topographic_similarity(
    objects: Sequence[tuple],
    messages: Sequence[list[int]],
    n_samples: int = 500,
    seed: int = 42,
) -> float:
    """
    Compute topographic similarity (topsim) between object space and message space.

    Args:
        objects  : list of attribute tuples, one per trial
        messages : list of byte lists (emergent messages), one per trial
        n_samples: number of random pairs to sample (full O(n^2) is slow)
        seed     : random seed
    Returns:
        Spearman correlation coefficient in [-1, 1]
        (higher = more compositional)
    """
    assert len(objects) == len(messages), "objects and messages must be same length"
    n = len(objects)
    rng = random.Random(seed)

    # Sample pairs
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > n_samples:
        pairs = rng.sample(all_pairs, n_samples)
    else:
        pairs = all_pairs

    attr_dists = []
    msg_dists = []
    for i, j in pairs:
        attr_dists.append(attribute_distance(objects[i], objects[j]))
        msg_dists.append(edit_distance(messages[i], messages[j]))

    if len(set(attr_dists)) < 2 or len(set(msg_dists)) < 2:
        return 0.0  # degenerate case

    corr, _ = spearmanr(attr_dists, msg_dists)
    if np.isnan(corr):
        return 0.0
    return float(corr)


# ---------------------------------------------------------------------------
# Positional disentanglement
# ---------------------------------------------------------------------------

def positional_disentanglement(
    objects: Sequence[tuple],
    messages: Sequence[list[int]],
    max_pos: int = 8,
) -> dict[str, float]:
    """
    For each byte position p in the message and each attribute a,
    compute how much variance in byte[p] is explained by attribute a.

    Returns a dict: "pos_{p}_attr_{a}" → mutual information proxy (eta^2).
    High value means position p is mostly determined by attribute a.
    """
    from collections import defaultdict

    results = {}
    n_attrs = len(objects[0]) if objects else 0

    # Pad messages to max_pos
    padded = [m[:max_pos] + [-1] * max(0, max_pos - len(m)) for m in messages]

    for pos in range(max_pos):
        byte_at_pos = [padded[i][pos] for i in range(len(padded)) if padded[i][pos] >= 0]
        if not byte_at_pos:
            continue

        for attr_idx in range(n_attrs):
            attr_vals = [objects[i][attr_idx] for i in range(len(padded)) if padded[i][pos] >= 0]

            # Eta-squared: proportion of variance in byte_at_pos explained by attr grouping
            byte_arr = np.array(byte_at_pos, dtype=float)
            overall_mean = byte_arr.mean()
            ss_total = ((byte_arr - overall_mean) ** 2).sum()

            if ss_total == 0:
                results[f"pos{pos}_attr{attr_idx}"] = 0.0
                continue

            groups = defaultdict(list)
            for v, b in zip(attr_vals, byte_at_pos):
                groups[v].append(b)

            ss_between = sum(
                len(grp) * (np.mean(grp) - overall_mean) ** 2
                for grp in groups.values()
            )
            results[f"pos{pos}_attr{attr_idx}"] = ss_between / ss_total

    return results


# ---------------------------------------------------------------------------
# Convenience: run all compositionality metrics
# ---------------------------------------------------------------------------

def evaluate_compositionality(
    objects: Sequence[tuple],
    messages: Sequence[list[int]],
    n_topsim_samples: int = 500,
    seed: int = 42,
) -> dict:
    topsim = topographic_similarity(objects, messages, n_topsim_samples, seed)
    pos_disent = positional_disentanglement(objects, messages)

    # Summary: max disentanglement per attribute
    n_attrs = len(objects[0]) if objects else 0
    max_disent_per_attr = {}
    for attr_idx in range(n_attrs):
        vals = [v for k, v in pos_disent.items() if f"attr{attr_idx}" in k]
        max_disent_per_attr[f"attr{attr_idx}_max_disentanglement"] = max(vals) if vals else 0.0

    return {
        "topsim": topsim,
        **max_disent_per_attr,
        "positional_disentanglement": pos_disent,
    }
