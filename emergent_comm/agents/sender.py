"""
Sender agent.

Observes an object (color, shape, size) and produces a byte-stream
message using the hierarchical decoder. The sender has a small
attribute-to-embedding network that maps object attributes into
an initial "context" embedding, which seeds the backbone as a
single word embedding (the object's identity).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AgentConfig, EncoderConfig, BackboneConfig, DecoderConfig
from ..models.decoder import CharacterDecoder, EOS_TOKEN
from ..models.backbone import WordBackbone


class ObjectEmbedder(nn.Module):
    """
    Maps discrete object attributes (color, shape, size) to a
    continuous embedding in backbone space.

    Each attribute gets its own lookup table; their embeddings are
    summed and projected.
    """

    def __init__(self, n_colors: int, n_shapes: int, n_sizes: int, d_model: int):
        super().__init__()
        d_attr = d_model // 3 + 1  # roughly equal split
        self.color_emb = nn.Embedding(n_colors, d_attr)
        self.shape_emb = nn.Embedding(n_shapes, d_attr)
        self.size_emb = nn.Embedding(n_sizes, d_attr)
        self.proj = nn.Linear(3 * d_attr, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attrs: [B, 3]  — (color_idx, shape_idx, size_idx)
        Returns:
            emb  : [B, d_model]
        """
        c = self.color_emb(attrs[:, 0])
        s = self.shape_emb(attrs[:, 1])
        z = self.size_emb(attrs[:, 2])
        return self.norm(self.proj(torch.cat([c, s, z], dim=-1)))


class Sender(nn.Module):
    """
    Sender agent: object → byte message.

    Architecture:
        1. ObjectEmbedder maps target attributes → word embedding e^0
        2. Backbone(e^0) → predictive embedding p^0
        3. Decoder generates bytes of the message word by word
           until EOS or max_message_len

    For simplicity in this signaling game, the message is a single
    "word" (one pass through the decoder). Multi-word messages are
    a natural extension.
    """

    def __init__(
        self,
        n_colors: int,
        n_shapes: int,
        n_sizes: int,
        bb_cfg: BackboneConfig,
        dec_cfg: DecoderConfig,
        agent_cfg: AgentConfig,
    ):
        super().__init__()
        self.agent_cfg = agent_cfg
        self.object_embedder = ObjectEmbedder(n_colors, n_shapes, n_sizes, bb_cfg.d_model)
        self.backbone = WordBackbone(bb_cfg)
        self.decoder = CharacterDecoder(dec_cfg, d_backbone=bb_cfg.d_model)

    def forward(
        self,
        target_attrs: torch.Tensor,
        target_bytes: torch.Tensor | None = None,
        target_pad_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            target_attrs    : [B, 3]         — object attributes
            target_bytes    : [B, msg_len]   — teacher-forced bytes (training only)
            target_pad_mask : [B, msg_len]   — True where padded (training only)
        Returns dict with:
            logits          : [B, msg_len, vocab_size]  (training)
            pred_emb        : [B, d_backbone]
            message_bytes   : [B, gen_len]              (inference)
            log_probs       : [B, gen_len]              (inference)
        """
        obj_emb = self.object_embedder(target_attrs)            # [B, d_bb]
        pred_emb = self.backbone(obj_emb.unsqueeze(1))[:, 0, :] # [B, d_bb]

        result = {"pred_emb": pred_emb}

        if target_bytes is not None:
            # Teacher-forced training path
            logits = self.decoder(pred_emb, target_bytes, key_padding_mask=target_pad_mask)
            result["logits"] = logits
        else:
            # Autoregressive inference path (fixed-length to prevent EOS collapse)
            msg_bytes, log_probs, logits, soft_tokens = self.decoder.generate(
                pred_emb,
                temperature=self.agent_cfg.temperature,
                max_len=self.agent_cfg.max_message_len,
                fixed_length=True,
                use_straight_through=self.agent_cfg.use_straight_through,
            )
            result["message_bytes"] = msg_bytes
            result["log_probs"] = log_probs
            result["logits"] = logits
            if soft_tokens is not None:
                result["message_soft"] = soft_tokens

        return result
