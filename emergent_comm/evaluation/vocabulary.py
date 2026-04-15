"""
Vocabulary analysis for emergent communication.

After training, we pass many objects through the sender and collect
the emitted byte streams. The entropy-adaptive segmenter then splits
each stream into "words". We analyze:

  - How many unique emergent words exist?
  - What is the frequency distribution? (Zipfian → more natural)
  - Do specific words correlate with specific attributes?
  - How stable is the vocabulary across random seeds?
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import torch

from ..models.segmentation import EntropySegmenter
from ..config import SegmentationConfig


def bytes_to_str(byte_list: list[int]) -> str:
    """Convert a byte list to a printable string for display."""
    try:
        return bytes(b for b in byte_list if 32 <= b < 127).decode("ascii") or f"<{byte_list}>"
    except Exception:
        return str(byte_list)


def collect_vocabulary(
    objects: Sequence[tuple],
    messages: Sequence[list[int]],
    seg_cfg: SegmentationConfig,
    logits_list: Sequence[torch.Tensor] | None = None,
) -> dict:
    """
    Segment all messages and build a vocabulary frequency table.

    Args:
        objects     : list of attribute tuples
        messages    : list of flat byte lists (unsegmented messages)
        seg_cfg     : segmentation configuration
        logits_list : list of [msg_len, vocab_size] logit tensors (for entropy segmentation)
                      If None, fall back to whitespace-style fixed splitting.
    Returns:
        dict with vocabulary analysis results
    """
    segmenter = EntropySegmenter(
        threshold=seg_cfg.entropy_threshold,
        min_len=seg_cfg.min_segment_len,
        max_len=seg_cfg.max_segment_len,
        smoothing_window=seg_cfg.smoothing_window,
    )

    all_words: list[tuple[int, ...]] = []
    word_to_objects: defaultdict[tuple, list[tuple]] = defaultdict(list)

    for i, (obj, msg) in enumerate(zip(objects, messages)):
        if logits_list is not None and i < len(logits_list):
            segments = segmenter.segment_with_logits(msg, logits_list[i])
        else:
            # Fallback: treat whole message as one word
            segments = [msg]

        for seg in segments:
            word = tuple(seg)
            all_words.append(word)
            word_to_objects[word].append(obj)

    freq = Counter(all_words)
    vocab_size = len(freq)
    total_tokens = len(all_words)

    # Zipf fit: rank * frequency should be roughly constant
    ranks = list(range(1, min(vocab_size + 1, 101)))
    sorted_freqs = [f for _, f in freq.most_common(100)]
    
    zipf_coeff = 0.0
    if len(sorted_freqs) > 1:
        log_ranks = np.log(ranks[:len(sorted_freqs)])
        log_freqs = np.log(np.array(sorted_freqs) + 1e-9) # Use small epsilon
        if np.var(log_freqs) > 1e-12:
            zipf_coeff = float(np.corrcoef(log_ranks, log_freqs)[0, 1])
            if np.isnan(zipf_coeff):
                zipf_coeff = 0.0
    
    # Attribute alignment: for each top word, which attribute value is most associated?
    top_words = freq.most_common(20)
    word_attribute_alignment = {}
    for word, _ in top_words:
        objs = word_to_objects[word]
        if not objs:
            continue
        n_attrs = len(objs[0])
        dominant_attrs = []
        for attr_idx in range(n_attrs):
            vals = [o[attr_idx] for o in objs]
            most_common_val = Counter(vals).most_common(1)[0]
            purity = most_common_val[1] / len(vals)
            dominant_attrs.append((attr_idx, most_common_val[0], purity))
        word_attribute_alignment[bytes_to_str(list(word))] = dominant_attrs

    return {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "type_token_ratio": vocab_size / max(total_tokens, 1),
        "zipf_correlation": zipf_coeff,
        "top_words": [(bytes_to_str(list(w)), c) for w, c in freq.most_common(20)],
        "word_attribute_alignment": word_attribute_alignment,
        "frequency_distribution": dict(freq.most_common(50)),
    }


def print_vocabulary_report(vocab_results: dict):
    """Pretty-print vocabulary analysis results."""
    print(f"\n{'=' * 60}")
    print(f"EMERGENT VOCABULARY REPORT")
    print(f"{'=' * 60}")
    print(f"Vocabulary size    : {vocab_results['vocab_size']}")
    print(f"Total tokens       : {vocab_results['total_tokens']}")
    print(f"Type-token ratio   : {vocab_results['type_token_ratio']:.4f}")
    print(f"Zipf correlation   : {vocab_results['zipf_correlation']:.4f}  (higher = more Zipfian)")
    print(f"\nTop 20 emergent words:")
    for word_str, count in vocab_results["top_words"]:
        alignment = vocab_results["word_attribute_alignment"].get(word_str, [])
        align_str = ", ".join(f"attr{a}={v}({p:.0%})" for a, v, p in alignment)
        print(f"  {word_str!r:20s}  count={count:>5}  [{align_str}]")
    print(f"{'=' * 60}\n")
