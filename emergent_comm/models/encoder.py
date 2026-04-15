"""
Character-level encoder (E in the paper).

Takes a batch of byte sequences (one per word segment) and produces
a single word embedding per segment by reading the output at the
prepended [W] token position.

Architecture: bidirectional transformer over bytes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import EncoderConfig

# Special byte values (must match AgentConfig)
W_TOKEN = 2   # prepended to every word; its output becomes the word embedding
PAD_TOKEN = 3


class ByteEmbedding(nn.Module):
    """Embeds byte values + adds sinusoidal positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.register_buffer("pos_enc", self._make_pos_enc(max_len, d_model))

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        emb = self.token_emb(x)                       # [B, L, d]
        emb = emb + self.pos_enc[:, : x.size(1)]      # broadcast over batch
        return emb

    def forward_soft(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """Differentiable embedding for soft one-hot distributions.

        Args:
            soft_tokens: [B, L, vocab_size]
        Returns:
            embeddings : [B, L, d_model]
        """
        emb = soft_tokens @ self.token_emb.weight      # [B, L, d_model]
        emb = emb + self.pos_enc[:, : soft_tokens.size(1)]
        return emb


class CharacterEncoder(nn.Module):
    """
    Bidirectional transformer encoder that maps a byte sequence
    representing one word into a single fixed-size word embedding.

    Input : byte_ids  [B, 1 + word_len]  (first token is always W_TOKEN)
    Output: word_embs [B, d_backbone]
    """

    def __init__(self, cfg: EncoderConfig, d_backbone: int):
        super().__init__()
        self.cfg = cfg
        self.embedding = ByteEmbedding(cfg.vocab_size, cfg.d_model, max_len=128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Project from character dim to backbone dim
        self.proj = nn.Linear(cfg.d_model, d_backbone)

    def forward(self, byte_ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            byte_ids          : [B, seq_len]  — first position is W_TOKEN
            key_padding_mask  : [B, seq_len]  — True where padded (optional)
        Returns:
            word_embeddings   : [B, d_backbone]
        """
        x = self.embedding(byte_ids)                                   # [B, L, d]
        x = self.transformer(x, src_key_padding_mask=key_padding_mask) # [B, L, d]
        w_output = x[:, 0, :]                                          # [W] token position
        return self.proj(w_output)                                     # [B, d_backbone]

    def forward_soft(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """Encode soft token distributions (straight-through training path).

        Args:
            soft_tokens: [B, seq_len, vocab_size] — first position is W_TOKEN one-hot
        Returns:
            word_embeddings: [B, d_backbone]
        """
        x = self.embedding.forward_soft(soft_tokens)  # [B, L, d_char]
        x = self.transformer(x)                        # [B, L, d_char]
        w_output = x[:, 0, :]                          # [W] token position
        return self.proj(w_output)                     # [B, d_backbone]
