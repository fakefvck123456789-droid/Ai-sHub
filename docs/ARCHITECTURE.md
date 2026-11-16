# Ai'Hub architecture — foundation stage

## Scope

Ai'Hub is intended to build and run its own models and tooling. The repository currently contains a small statistical baseline, an experimental neural language model implementation, a local inference API, and a mobile-first PWA. It is not a hosted model service and does not forward prompts to third-party AI APIs.

## Implemented pipeline

1. Read a developer-supplied UTF-8 corpus.
2. Split text into training and held-out validation segments.
3. Train a deterministic byte-level BPE tokenizer on the training segment only.
4. Train a decoder-only Transformer with next-token cross-entropy.
5. Report training and validation loss and save a checkpoint with model configuration and tokenizer merges.
6. Load the checkpoint for local CLI generation or the local HTTP API.
7. Use the PWA in `app/` to send prompts to the loopback-only API.

The earlier bigram pipeline remains available as a dependency-light smoke-test baseline. It is not a neural network and has no semantic reasoning ability.

## Core 0.1 architecture

The implemented Transformer includes:
- Token embedding and tied output projection
- RMSNorm
- Multi-head causal self-attention with RoPE
- SwiGLU feed-forward blocks
- Residual connections
- Next-token cross-entropy objective
- Temperature and top-k autoregressive generation

The canonical configuration is in `configs/core_0_1.json`: 4 layers, 4 attention heads, 256 embedding dimensions, 1024 feed-forward dimensions, and 512 context positions. Shorter context lengths are supported for smoke tests.

## Runtime components

- `core/transformer_generate.py` — local checkpoint generation CLI
- `core/transformer_server.py` — local HTTP API, bound to 127.0.0.1 by default
- `app/` — installable PWA shell, with local-only CORS origins configured for port 8080
- `.github/workflows/tests.yml` — automated Python test suite

## Known limitations

- The trainer is a minimal single-corpus experiment. It has held-out validation but no checkpoint resume, learning-rate scheduler, mixed precision, gradient accumulation, or distributed training.
- Byte-level BPE is a simple reference implementation, not optimized for very large corpora.
- No curated real-world training corpus or pretrained weights are bundled.
- No benchmark results or model-quality claims are available.
- The PWA is not a compiled native Android APK.
- PyTorch support on Android/Termux varies; mobile training and inference have not been verified on-device.

## Runtime security

The Transformer HTTP server binds to localhost by default and restricts browser CORS to the PWA's local origins. Keep it on loopback. Do not expose it to a public network without authentication, stronger request controls, and a security review.
