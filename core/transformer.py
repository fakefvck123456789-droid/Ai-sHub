"""Small decoder-only Transformer for Ai'Hub Core 0.1.

This is an implementation of the architecture, not pretrained weights.
The default configuration is intentionally small enough for CPU experiments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int = 512
    layers: int = 4
    heads: int = 4
    embedding_dim: int = 256
    feed_forward_dim: int = 1024
    dropout: float = 0.0

    def __post_init__(self):
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.context_length < 1 or self.layers < 1 or self.heads < 1:
            raise ValueError("context_length, layers, and heads must be positive")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.embedding_dim // self.heads % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.feed_forward_dim < 1:
            raise ValueError("feed_forward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict) -> "TransformerConfig":
        return cls(**values)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to [batch, heads, time, head_dim]."""
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.embedding_dim // config.heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, time, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        positions = torch.arange(time, device=x.device)
        q, k = apply_rope(q, positions), apply_rope(k, positions)

        # Explicit causal attention uses standard tensor operations so the
        # model can be exported to ONNX Runtime on Android and other devices.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones((time, time), device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(config.embedding_dim, config.feed_forward_dim, bias=False)
        self.value = nn.Linear(config.embedding_dim, config.feed_forward_dim, bias=False)
        self.output = nn.Linear(config.feed_forward_dim, config.embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(x)) * self.value(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.embedding_dim)
        self.feed_forward = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x


class AiHubTransformer(nn.Module):
    """Autoregressive language model; input and output are integer token IDs."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.final_norm = RMSNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds configured context_length")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be non-negative and temperature positive")
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("prompt must be a non-empty [batch, time] tensor")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.config.context_length:]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    if top_k < 1:
                        raise ValueError("top_k must be positive")
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                token_ids = torch.cat((token_ids, next_id), dim=1)
        finally:
            self.train(was_training)
        return token_ids

    def config_dict(self) -> dict:
        return asdict(self.config)
