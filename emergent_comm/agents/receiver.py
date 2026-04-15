"""
Receiver agent.

Reads the sender's byte-stream message through the hierarchical
encoder → backbone pipeline, then scores each candidate object
against the resulting message embedding. Predicts which candidate
the sender was referring to.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EncoderConfig, BackboneConfig
from ..models.encoder import CharacterEncoder, W_TOKEN, PAD_TOKEN
from ..models.backbone import WordBackbone


class Receiver(nn.Module):
    """
    Receiver agent: byte message + candidates → predicted target index.

    Architecture:
        1. CharacterEncoder reads message bytes → message embedding
        2. WordBackbone (optional; useful for multi-word messages) → context
        3. Each candidate object is embedded by a shared ObjectEmbedder
        4. Dot-product similarity between message embedding and each
           candidate embedding → softmax → predicted target distribution
    """

    def __init__(
        self,
        n_colors: int,
        n_shapes: int,
        n_sizes: int,
        enc_cfg: EncoderConfig,
        bb_cfg: BackboneConfig,
    ):
        super().__init__()
        self.encoder = CharacterEncoder(enc_cfg, d_backbone=bb_cfg.d_model)
        self.backbone = WordBackbone(bb_cfg)

        # Candidate object embedder (same structure as sender's ObjectEmbedder)
        d_attr = bb_cfg.d_model // 3 + 1
        self.color_emb = nn.Embedding(n_colors, d_attr)
        self.shape_emb = nn.Embedding(n_shapes, d_attr)
        self.size_emb = nn.Embedding(n_sizes, d_attr)
        self.cand_proj = nn.Linear(3 * d_attr, bb_cfg.d_model)
        self.cand_norm = nn.LayerNorm(bb_cfg.d_model)

        # Scale dot products (learnable temperature)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def embed_candidates(self, candidate_attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_attrs: [B, n_cands, 3]
        Returns:
            cand_embs      : [B, n_cands, d_model]
        """
        B, N, _ = candidate_attrs.shape
        flat = candidate_attrs.view(B * N, 3)
        c = self.color_emb(flat[:, 0])
        s = self.shape_emb(flat[:, 1])
        z = self.size_emb(flat[:, 2])
        emb = self.cand_norm(self.cand_proj(torch.cat([c, s, z], dim=-1)))
        return emb.view(B, N, -1)

    def encode_message(
        self,
        message_bytes: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode byte message into a single embedding vector.

        Args:
            message_bytes: [B, msg_len]  — bytes with W_TOKEN prepended
            pad_mask     : [B, msg_len]  — True where padded
        Returns:
            msg_emb: [B, d_backbone]
        """
        # Encoder reads the message as a single "word"
        word_emb = self.encoder(message_bytes, key_padding_mask=pad_mask)  # [B, d_bb]
        # Backbone over a single word (trivial but keeps architecture consistent)
        pred_emb = self.backbone(word_emb.unsqueeze(1))[:, 0, :]           # [B, d_bb]
        return pred_emb

    def forward(
        self,
        message_bytes: torch.Tensor,
        candidate_attrs: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            message_bytes  : [B, msg_len]       — bytes (W_TOKEN prepended)
            candidate_attrs: [B, n_cands, 3]
            pad_mask       : [B, msg_len]        — True where padded
        Returns dict with:
            logits         : [B, n_cands]        — unnormalized scores
            log_probs      : [B, n_cands]        — log-softmax scores
            predicted_idx  : [B]                 — argmax prediction
            msg_embedding  : [B, d_backbone]     — for analysis
        """
        msg_emb = self.encode_message(message_bytes, pad_mask)   # [B, d_bb]
        cand_embs = self.embed_candidates(candidate_attrs)        # [B, N, d_bb]

        # Scaled dot product: [B, N]
        scale = self.log_scale.exp()
        scores = torch.bmm(cand_embs, msg_emb.unsqueeze(-1)).squeeze(-1) * scale

        log_probs = F.log_softmax(scores, dim=-1)
        predicted_idx = scores.argmax(dim=-1)

        return {
            "logits": scores,
            "log_probs": log_probs,
            "predicted_idx": predicted_idx,
            "msg_embedding": msg_emb,
        }
