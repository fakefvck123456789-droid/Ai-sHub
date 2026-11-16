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

## Local model families

The local router accepts registered checkpoints with explicit capability labels and validates tokenizer/model vocabulary compatibility. The Core text checkpoint is currently the only shipped model. A future Code, Vision, Image, Voice, or Motion model must provide its own trained checkpoint and a real implementation of its declared capability before it is registered; a capability label alone is not a model.

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

## Training controls

The trainer supports gradient accumulation, linear warmup, and cosine decay to a configurable minimum learning-rate ratio:

```bash
python -m core.transformer_train --input data/prepared/corpus.txt \
  --output artifacts/core-0.1.pt --steps 10000 --context-length 128 \
  --batch-size 2 --gradient-accumulation-steps 4 \
  --warmup-steps 300 --min-lr-ratio 0.1
```

The effective batch size is `batch-size × gradient-accumulation-steps`. The scheduler's `--steps` value is the target total optimizer-step count, including when resuming. The saved checkpoint records the effective batch size and scheduler settings.

## Local performance benchmark

After training a checkpoint, measure generation latency and throughput on the machine where it will run:

```bash
python -m core.transformer_benchmark --checkpoint artifacts/core-0.1.pt \
  --prompt "Ai'Hub" --new-tokens 32 --warmup 1 --repeats 5 --device cpu
```

The report includes mean/median latency, generated tokens per second, parameter count, and process peak RSS where the platform exposes it. This is a local measurement, not a published device benchmark.

## Known gaps

- The trainer uses a deterministic tail slice as held-out validation and reports validation loss during training. The standalone `core.transformer_eval` CLI computes next-token loss and perplexity on a separate corpus. It supports checkpoint resume with optimizer and RNG state, and verifies the original corpus and validation split. Gradient accumulation and warmup-plus-cosine learning-rate decay are implemented; mixed precision and distributed training remain future work.
- No large, curated, legally documented corpus is bundled.
- No pretrained weights or benchmark results are supplied.
- Mobile PyTorch availability is platform-dependent; CPU training on an Android phone is not guaranteed.
