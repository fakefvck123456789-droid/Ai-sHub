# Ai'Hub

An independent, research-first AI software and model ecosystem. Ai'Hub is designed to develop and run its own models and tooling rather than proxying prompts to third-party AI APIs.

## Current status

**Foundation stage — Ai'Hub Core 0.1 is not pretrained.**

The first milestone is a reproducible, CPU-friendly training and inference pipeline. The initial n-gram prototype validates data loading, tokenization, training, and generation. It is explicitly a pipeline test, not a Transformer or a general reasoning model.

## Repository layout

- `core/` — tokenizer, prototype model, training, inference, and local API server
- `data/` — corpus instructions and provenance template; no unlicensed corpus is bundled
- `tests/` — automated tests
- `docs/` — architecture and development notes
- `scripts/` — setup and smoke-test helpers

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m core.train --input data/sample.txt --output artifacts/ngram.json
python -m core.server --model artifacts/ngram.json
```

The sample corpus is synthetic and included only to verify the pipeline. Before training on other text, confirm that its license permits the intended use and record its source in `data/CORPUS_SOURCES.md`.

## Android / Termux

The prototype is CPU-only and intentionally small. On Android, install Python in Termux, clone this repository, install requirements, and run the same smoke test. Training the planned Transformer is a later milestone and may be slow or memory-constrained on mobile hardware.

## Planned Core 0.1 architecture

- Decoder-only Transformer
- 512-token context
- 4 layers, 4 attention heads
- 256-dimensional embeddings, 1024-dimensional feed-forward blocks
- RoPE, RMSNorm, causal self-attention
- Next-token prediction objective
- Custom BPE tokenizer (planned; current prototype uses a simple word tokenizer)

No pretrained weights are included. No claim of independent reasoning capability is made at this stage.

## License

The repository's software is licensed under the MIT License. This does not grant rights to third-party datasets; each dataset must retain its own license and provenance.
