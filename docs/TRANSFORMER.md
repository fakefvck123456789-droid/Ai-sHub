# Ai'Hub Core 0.1 — Transformer implementation

## Implemented

The current implementation in `core/transformer.py` is a decoder-only autoregressive Transformer:

- Token embedding with tied output projection
- Pre-normalized residual blocks
- RMSNorm
- Multi-head causal self-attention using PyTorch scaled-dot-product attention
- Rotary position embeddings (RoPE)
- SwiGLU feed-forward network
- Cross-entropy next-token objective
- Temperature and top-k autoregressive sampling

The target architecture is configured in `configs/core_0_1.json`: 4 blocks, 4 heads, 256 embedding dimensions, 1024 feed-forward dimensions, and 512 context positions. The training CLI allows a shorter context for smoke tests and constrained experiments.

## Tokenizer

`core/bpe_tokenizer.py` implements byte-level BPE. The 256 base IDs represent individual UTF-8 bytes; learned merge IDs start at 256. This makes unseen scripts and symbols representable without an unknown-token fallback. It uses ranked heap-based merging for encoding in 8 KB chunks and learns merges from a bounded, evenly spaced sample of up to 100 KB by default (adjustable with `--bpe-training-bytes`). It remains a reference implementation, not an optimized production tokenizer.

## Train a smoke-test checkpoint

Install a compatible PyTorch build, then run:

```bash
python -m unittest discover -s tests -v
python -m core.transformer_train \
  --input data/sample.txt \
  --output artifacts/core-0.1-smoke.pt \
  --steps 2 --batch-size 2 --context-length 16 --vocab-size 256
python -m core.transformer_generate \
  --checkpoint artifacts/core-0.1-smoke.pt \
  --prompt "Ai'Hub" --max-new-tokens 20
```

The included synthetic corpus is for pipeline verification only. It is not a meaningful language-model training dataset. A tiny training loss or generated string does not establish useful model quality.

## Known gaps

- The trainer uses a deterministic tail slice as held-out validation and reports validation loss during training. The standalone `core.transformer_eval` CLI computes next-token loss and perplexity on a separate corpus. It supports checkpoint resume with optimizer and RNG state, and verifies the original corpus and validation split. Learning-rate scheduling, mixed precision, gradient accumulation, and distributed training are not implemented yet.
- No large, curated, legally documented corpus is bundled.
- No pretrained weights or benchmark results are supplied.
- Mobile PyTorch availability is platform-dependent; CPU training on an Android phone is not guaranteed.
