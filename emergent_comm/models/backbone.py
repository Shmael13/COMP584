"""
Word-level backbone (B in the paper).

Takes a sequence of word embeddings produced by the encoder and
processes them with a causal transformer, producing predictive
word embeddings used by the decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import BackboneConfig


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d]
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)


class WordBackbone(nn.Module):
    """
    Causal transformer that maps a sequence of word embeddings
    [e^1, ..., e^L] to predictive word embeddings [p^1, ..., p^L].

    p^i is used by the decoder to generate the characters of word i+1.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = LearnedPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        # Using TransformerDecoder with no cross-attention memory gives causal self-attention
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future words."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_embeddings: [B, S, d_backbone]
        Returns:
            predictive_embeddings: [B, S, d_backbone]
        """
        x = self.pos_enc(word_embeddings)
        x = self.input_norm(x)
        mask = self._causal_mask(x.size(1), x.device)
        # Pass x as both tgt and memory; the causal mask blocks future positions
        x = self.transformer(tgt=x, memory=x, tgt_mask=mask, memory_mask=mask)
        return self.output_norm(x)
