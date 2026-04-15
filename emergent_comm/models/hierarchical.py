"""
Full hierarchical autoregressive model.

Wires together Encoder → Backbone → Decoder and exposes
a clean interface for both training and inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import EncoderConfig, BackboneConfig, DecoderConfig
from .encoder import CharacterEncoder, W_TOKEN, PAD_TOKEN
from .backbone import WordBackbone
from .decoder import CharacterDecoder


class HierarchicalModel(nn.Module):
    """
    Hierarchical autoregressive transformer.

    Encodes a sequence of word byte-sequences → word embeddings,
    runs them through a causal backbone, and decodes each predictive
    embedding back into the next word's bytes.
    """

    def __init__(self, enc_cfg: EncoderConfig, bb_cfg: BackboneConfig, dec_cfg: DecoderConfig):
        super().__init__()
        self.encoder = CharacterEncoder(enc_cfg, d_backbone=bb_cfg.d_model)
        self.backbone = WordBackbone(bb_cfg)
        self.decoder = CharacterDecoder(dec_cfg, d_backbone=bb_cfg.d_model)

    def encode_words(
        self,
        word_byte_ids: torch.Tensor,
        word_pad_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a batch of word sequences.

        Args:
            word_byte_ids : [B, S, word_len]  — byte ids per word, W_TOKEN prepended
            word_pad_masks: [B, S, word_len]  — True where padded
        Returns:
            word_embeddings: [B, S, d_backbone]
        """
        B, S, L = word_byte_ids.shape
        flat_ids = word_byte_ids.view(B * S, L)
        flat_masks = word_pad_masks.view(B * S, L) if word_pad_masks is not None else None

        flat_embs = self.encoder(flat_ids, key_padding_mask=flat_masks)  # [B*S, d_backbone]
        return flat_embs.view(B, S, -1)

    def forward(
        self,
        word_byte_ids: torch.Tensor,
        word_pad_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            word_byte_ids : [B, S, word_len]
            word_pad_masks: [B, S, word_len]
        Returns:
            logits        : [B, S, word_len, vocab_size]
            word_embeddings: [B, S, d_backbone]
        """
        B, S, L = word_byte_ids.shape

        # Encode all words → word embeddings
        word_embs = self.encode_words(word_byte_ids, word_pad_masks)   # [B, S, d_bb]

        # Run backbone over word sequence
        pred_embs = self.backbone(word_embs)                            # [B, S, d_bb]

        # Decode: p^i generates word i+1; shift by one
        # pred_embs[:, :-1] predicts the characters of word_byte_ids[:, 1:]
        pred = pred_embs[:, :-1].contiguous().view((B * (S - 1)), -1)  # [B*(S-1), d_bb]
        tgt = word_byte_ids[:, 1:].contiguous().view(B * (S - 1), L)   # [B*(S-1), L]
        masks = (
            word_pad_masks[:, 1:].contiguous().view(B * (S - 1), L)
            if word_pad_masks is not None else None
        )

        logits_flat = self.decoder(pred, tgt, key_padding_mask=masks)   # [B*(S-1), L, vocab]
        logits = logits_flat.view(B, S - 1, L, -1)

        return logits, word_embs

    def get_predictive_embedding(self, word_byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Given a sequence of observed words, return the backbone's
        predictive embedding for the *next* (unobserved) word.

        Args:
            word_byte_ids: [B, S, word_len]
        Returns:
            pred_emb: [B, d_backbone]
        """
        word_embs = self.encode_words(word_byte_ids)    # [B, S, d_bb]
        pred_embs = self.backbone(word_embs)             # [B, S, d_bb]
        return pred_embs[:, -1, :]                       # last position predicts next word
