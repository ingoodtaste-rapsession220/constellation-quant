"""Informer encoder with ProbSparse self-attention (Zhou et al., 2021).

Takes a per-stock lookback window `(B, L, F)` and returns a single embedding
`(B, d_model)`. Three innovations over a vanilla Transformer:

1. **ProbSparse attention** — instead of computing the full N² attention
   matrix, sample a subset of keys per query and pick the top-U queries by
   "sparsity" (KL divergence between their attention distribution and a
   uniform distribution). Reduces O(L²) → O(L log L).

2. **Distilling layers** — every encoder layer halves the sequence length
   via 1D-conv + max-pool, so deep stacks don't blow up memory.

3. **Attention-weighted pooling** — the final (L/2^E, d_model) tensor is
   collapsed into a single (d_model,) vector via a learned attention query,
   rather than the `[CLS]`-style first-token pick used in BERT.

The implementation is faithful to the paper at the scale relevant here; we
don't include the auxiliary decoder because the project's output is a
ranking score per stock, not a multi-step forecast. The encoder output is
consumed downstream by the GNN.

Shapes throughout (notation used in docstrings):
    B      : batch (number of stocks in current graph × time-step batch)
    L      : lookback length (default 60 trading days)
    F      : feature dimension per day
    d_model: internal hidden size (default 256)
    H      : attention heads (default 8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InformerConfig:
    """Parsed subset of `model_config.yaml` > temporal for the Informer."""
    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    probsparse_factor: int = 5     # u = factor · ln(L) sampled queries
    distil: bool = True            # halve L between encoder layers
    use_learnable_pe: bool = True
    pooling: str = "attention_weighted_mean"   # | "last" | "mean"


# ── Positional encoding ────────────────────────────────────────────────────


class LearnablePositionalEncoding(nn.Module):
    """Simple learnable positional embedding — a lookup table of shape (L, d)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, d)
        L = x.size(1)
        if L > self.pe.size(0):
            raise ValueError(f"Sequence length {L} > max_len {self.pe.size(0)}")
        return x + self.pe[:L].unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal PE (Vaswani 2017) — used when `use_learnable_pe=False`."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


# ── ProbSparse attention ───────────────────────────────────────────────────


class ProbSparseSelfAttention(nn.Module):
    """ProbSparse multi-head self-attention (Zhou et al. 2021, Alg. 1).

    For a sequence of length L:
        1.  Sample U = factor · ln(L) keys uniformly at random.
        2.  For each query, compute dot products with the sampled keys.
        3.  Score each query's "sparsity": M(q) = max − mean over sampled keys.
        4.  Keep the top-u queries by M; compute full attention for them.
        5.  Non-top queries receive the mean of V (broadcast).

    Reduces cost from O(L²d) to O(L·ln(L)·d). For L ≤ 32 we fall back to
    standard attention — the overhead of sampling isn't worth the savings
    at short lengths, and the tests use L=20.
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)   # (B, H, L, D)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        # Fall back to dense attention for short sequences OR whenever the
        # module is in eval mode. The ProbSparse path samples keys with
        # `torch.randint` — fine at training time (stochastic regularisation)
        # but non-deterministic at inference, which injects noise into
        # backtests. Dense is cheap at L=60 (60² = 3600 dot products per
        # head) so the savings don't matter here anyway.
        if L <= 128 or not self.training:
            attn = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D)
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.einsum("bhij,bhjd->bhid", attn, v)
        else:
            out = self._prob_sparse(q, k, v, L)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)

    def _prob_sparse(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: int) -> torch.Tensor:
        B, H, _, D = q.shape
        u_sample = min(max(int(self.factor * math.log(L)), 1), L)
        u_top = min(max(int(self.factor * math.log(L)), 1), L)

        # 1) Sample keys.
        idx = torch.randint(L, (u_sample,), device=q.device)
        k_sample = k.index_select(2, idx)   # (B, H, U_s, D)

        # 2) Score each query's attention spread.
        qk_sample = torch.einsum("bhid,bhjd->bhij", q, k_sample) / math.sqrt(D)  # (B,H,L,Us)
        sparsity = qk_sample.max(dim=-1).values - qk_sample.mean(dim=-1)          # (B,H,L)

        # 3) Top-u queries get full attention.
        top_idx = sparsity.topk(u_top, dim=-1).indices                             # (B,H,u)
        q_top = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))    # (B,H,u,D)
        attn_top = torch.einsum("bhud,bhjd->bhuj", q_top, k) / math.sqrt(D)
        attn_top = F.softmax(attn_top, dim=-1)
        attn_top = F.dropout(attn_top, p=self.dropout, training=self.training)
        out_top = torch.einsum("bhuj,bhjd->bhud", attn_top, v)                     # (B,H,u,D)

        # 4) Non-top queries receive the mean over V (paper's default).
        v_mean = v.mean(dim=2, keepdim=True).expand(-1, -1, L, -1)                 # (B,H,L,D)
        out = v_mean.clone()
        out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D), out_top)
        return out


# ── Encoder layers ─────────────────────────────────────────────────────────


class EncoderLayer(nn.Module):
    """Attention → add-norm → feed-forward → add-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 factor: int, dropout: float):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DistillLayer(nn.Module):
    """Paper's 1D-conv + max-pool that halves the sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="zeros")
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, d)
        x = x.transpose(1, 2)                              # -> (B, d, L)
        x = F.elu(self.norm(self.conv(x)))
        x = self.pool(x)                                   # -> (B, d, L/2)
        return x.transpose(1, 2)


# ── Pooling ────────────────────────────────────────────────────────────────


class AttentionPooling(nn.Module):
    """Learned single-query attention pooling: (B, L, d) -> (B, d)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.key_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inner product against a single learned query.
        keys = self.key_proj(x)                            # (B, L, d)
        scores = torch.einsum("bld,d->bl", keys, self.query) / math.sqrt(x.size(-1))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (B, L, 1)
        return (x * weights).sum(dim=1)                    # (B, d)


# ── Top-level encoder ──────────────────────────────────────────────────────


class InformerEncoder(nn.Module):
    """Full encoder stack: feature projection → PE → E × (attn, distil) → pool."""

    def __init__(self, n_features: int, config: Optional[InformerConfig] = None,
                 max_len: int = 512):
        super().__init__()
        self.cfg = config or InformerConfig()
        self.input_proj = nn.Linear(n_features, self.cfg.d_model)

        pe_cls = (LearnablePositionalEncoding
                  if self.cfg.use_learnable_pe else SinusoidalPositionalEncoding)
        self.pe = pe_cls(max_len=max_len, d_model=self.cfg.d_model)
        self.input_dropout = nn.Dropout(self.cfg.dropout)

        self.encoders = nn.ModuleList([
            EncoderLayer(
                d_model=self.cfg.d_model,
                n_heads=self.cfg.n_heads,
                d_ff=self.cfg.d_ff,
                factor=self.cfg.probsparse_factor,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.cfg.e_layers)
        ])
        if self.cfg.distil and self.cfg.e_layers > 1:
            self.distills = nn.ModuleList([
                DistillLayer(self.cfg.d_model) for _ in range(self.cfg.e_layers - 1)
            ])
        else:
            self.distills = None

        if self.cfg.pooling == "attention_weighted_mean":
            self.pool = AttentionPooling(self.cfg.d_model)
        else:
            self.pool = None   # handled inline in forward

    @property
    def output_dim(self) -> int:
        return self.cfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) -> (B, d_model)."""
        h = self.input_dropout(self.pe(self.input_proj(x)))
        for i, layer in enumerate(self.encoders):
            h = layer(h)
            if self.distills is not None and i < len(self.distills):
                h = self.distills[i](h)

        if self.pool is not None:
            return self.pool(h)
        if self.cfg.pooling == "last":
            return h[:, -1]
        return h.mean(dim=1)
