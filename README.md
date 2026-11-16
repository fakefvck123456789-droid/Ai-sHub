# Ai'Hub

An independent, research-first AI software and model ecosystem. Ai'Hub is designed to develop and run its own models and tooling rather than proxying prompts to third-party AI APIs.

## Current status

**Foundation stage — Ai'Hub Core 0.1 is implemented as an untrained model architecture.**

The repository contains two distinct paths:
- A dependency-light bigram baseline for testing the original data pipeline.
- A PyTorch decoder-only Transformer implementation with causal attention, RoPE, RMSNorm, SwiGLU, next-token loss, generation, and a basic training loop.

Neither path includes pretrained weights. A successful training run on a tiny sample corpus is only a software smoke test, not evidence of useful language ability or general reasoning.

## Repository layout

- `core/` — tokenizer, bigram baseline, Transformer, training, inference, and local API server
- `data/` — corpus instructions and provenance template; no unlicensed corpus is bundled
- `tests/` — automated tests
- `docs/` — architecture and development notes

## Setup

Python 3.10+ is recommended. Install a PyTorch build compatible with your operating system and Python version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The n-gram pipeline can be exercised with:

```bash
python -m core.train --input data/sample.txt --output artifacts/ngram.json
python -m core.server --model artifacts/ngram.json
```

## Train the Transformer

The training script currently uses the simple word tokenizer. BPE is not implemented yet. The sample corpus is synthetic and far too small to produce a useful language model; it is suitable only for a short code-path smoke test.

```bash
python -m core.transformer_train --input data/sample.txt --output artifacts/core-0.1.pt --steps 2 --batch-size 2 --context-length 16
```

For real experiments, supply a legally usable, sufficiently large corpus and record its origin and license in `data/CORPUS_SOURCES.md`. The default architecture is 4 layers, 4 heads, 256-dimensional embeddings, 1024-dimensional feed-forward layers, and a configurable context length. Training cost rises quickly with context, batch size, and corpus size.

The training script defaults to CPU. Do not assume that a CUDA build is available. The full default model and training workload may be slow or memory-intensive on phones.

## Android / Termux

The bigram prototype can run with standard Python. PyTorch installation support varies by Android device, Python version, and available package builds; `pip install -r requirements.txt` is not guaranteed to work in Termux. For Transformer training, use a supported Linux, Windows, or macOS CPU environment unless a compatible Android PyTorch build has been independently verified.

## Known limitations / next engineering work

- Tokenizer is currently word-based; a trained BPE tokenizer is still required.
- Transformer training is a minimal single-corpus loop: no validation split, scheduler, resume, mixed precision, or distributed training yet.
- There is no checkpoint-loading inference CLI for the Transformer yet.
- There are no pretrained weights and no curated training corpus.
- The sample text does not establish model quality.

## License

The repository's software is licensed under the MIT License. This does not grant rights to third-party datasets; each dataset must retain its own license and provenance.
