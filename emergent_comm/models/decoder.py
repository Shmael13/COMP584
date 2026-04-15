"""
Character-level decoder (D in the paper).

Given a predictive word embedding p^i from the backbone, generates
the byte sequence of word i+1 autoregressively.

During training: teacher-forced with the actual next word's bytes.
During inference: samples bytes one at a time until EOS or max_len.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DecoderConfig
from .encoder import ByteEmbedding, W_TOKEN, PAD_TOKEN

EOS_TOKEN = 0


class CharacterDecoder(nn.Module):
    """
    Causal transformer decoder that generates byte sequences.

    The predictive word embedding p^i is injected as a prefix
    (similar to a memory vector), prepended before the target bytes.
    """

    def __init__(self, cfg: DecoderConfig, d_backbone: int):
        super().__init__()
        self.cfg = cfg
        self.d_backbone = d_backbone

        # Project backbone embedding into character space
        self.word_proj = nn.Linear(d_backbone, cfg.d_model)
        self.embedding = ByteEmbedding(cfg.vocab_size, cfg.d_model, max_len=128)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        predictive_emb: torch.Tensor,
        target_bytes: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            predictive_emb : [B, d_backbone]      — p^i from backbone
            target_bytes   : [B, seq_len]          — byte ids of next word (incl. W_TOKEN prefix)
            key_padding_mask: [B, seq_len]         — True where padded
        Returns:
            logits         : [B, seq_len, vocab_size]
        """
        # Project word embedding into character space, treat as memory
        memory = self.word_proj(predictive_emb).unsqueeze(1)  # [B, 1, d_char]

        tgt = self.embedding(target_bytes)                     # [B, L, d_char]
        seq_len = tgt.size(1)
        causal_mask = self._causal_mask(seq_len, tgt.device)

        out = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )
        out = self.output_norm(out)
        return self.lm_head(out)                               # [B, L, vocab_size]

    def generate(
        self,
        predictive_emb: torch.Tensor,
        temperature: float = 1.0,
        max_len: int | None = None,
        fixed_length: bool = True,
        use_straight_through: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Autoregressive generation of one word's byte sequence.

        Args:
            predictive_emb     : [B, d_backbone]
            fixed_length       : if True, always generate exactly max_len bytes (mask EOS).
            use_straight_through: if True, use Gumbel-softmax straight-through instead of
                                  REINFORCE sampling. Returns differentiable soft tokens
                                  that can be passed directly to the receiver encoder.
        Returns:
            byte_ids   : [B, max_len]          — hard byte indices (always)
            log_probs  : [B, max_len]          — log-probs for REINFORCE (zeros if ST)
            logits     : [B, max_len, vocab]   — raw logits at each step
            soft_tokens: [B, max_len, vocab]   — Gumbel one-hots (ST mode only, else None)
        """
        max_len = max_len or self.cfg.max_word_len
        B = predictive_emb.size(0)
        device = predictive_emb.device

        generated = torch.full((B, 1), W_TOKEN, dtype=torch.long, device=device)
        log_probs: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        soft_tokens_list: list[torch.Tensor] = []
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            logits = self.forward(predictive_emb, generated)  # [B, t, vocab]
            next_logits = logits[:, -1, :]                     # [B, vocab]
            all_logits.append(next_logits)

            scaled_logits = next_logits / temperature

            if fixed_length:
                scaled_logits = scaled_logits.clone()
                scaled_logits[:, EOS_TOKEN] = float("-inf")

            if use_straight_through:
                if self.training:
                    # Gumbel-softmax: hard one-hot forward, soft gradient backward
                    soft = F.gumbel_softmax(scaled_logits, tau=1.0, hard=True)
                else:
                    # Deterministic argmax during evaluation
                    idx = scaled_logits.argmax(dim=-1, keepdim=True)
                    soft = torch.zeros_like(scaled_logits).scatter_(-1, idx, 1.0)
                soft_tokens_list.append(soft)
                next_byte = soft.detach().argmax(dim=-1)
                log_probs.append(torch.zeros(B, device=device))
            else:
                probs = F.softmax(scaled_logits, dim=-1)
                next_byte = torch.multinomial(probs, num_samples=1).squeeze(-1)

                if not fixed_length:
                    next_byte = next_byte.masked_fill(done, EOS_TOKEN)

                log_prob = torch.log(probs.gather(1, next_byte.unsqueeze(1)).squeeze(1) + 1e-9)
                log_probs.append(log_prob)

            generated = torch.cat([generated, next_byte.detach().unsqueeze(1)], dim=1)

            if not fixed_length and not use_straight_through:
                done = done | (next_byte == EOS_TOKEN)
                if done.all():
                    break

        soft_out = torch.stack(soft_tokens_list, dim=1) if use_straight_through else None

        return (
            generated[:, 1:],
            torch.stack(log_probs, dim=1),
            torch.stack(all_logits, dim=1),
            soft_out,
        )
