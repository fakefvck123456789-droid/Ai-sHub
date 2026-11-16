# Ai'Hub

An independent, research-first AI software and model ecosystem. Ai'Hub is designed to develop and run its own models and tooling rather than proxying prompts to third-party AI APIs.

## Current status

See [implementation status and release gates](docs/STATUS.md) for what is implemented and what remains before a real Core model release.

**Foundation stage — Ai'Hub Core 0.1 architecture and training pipeline are implemented; no trained release model exists.**

The repository contains:
- A dependency-light bigram baseline for testing the original data pipeline.
- A PyTorch decoder-only Transformer with causal attention, RoPE, RMSNorm, SwiGLU, next-token loss, and autoregressive generation.
- A custom byte-level BPE tokenizer.
- A training and checkpoint inference CLI.
- A dynamic-sequence ONNX export path with ONNX Runtime numerical parity checks.

Neither model path includes pretrained weights. A successful run on the tiny sample corpus is only a software smoke test, not evidence of useful language ability or general reasoning.

## Repository layout

- `core/` — tokenizer, bigram baseline, Transformer, training, generation, export, and local API server
- `configs/` — canonical Core 0.1 architecture configuration
- `data/` — corpus instructions and provenance template; no unlicensed corpus is bundled
- `tests/` — automated tests
- `docs/` — architecture and development notes

## Setup

Python 3.10+ is recommended. Install a PyTorch build compatible with your operating system and Python version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux CPU build:
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m unittest discover -s tests -v
```

The n-gram pipeline can be exercised with:

```bash
python -m core.train --input data/sample.txt --output artifacts/ngram.json
python -m core.server --model artifacts/ngram.json
```

## Prepare a licensed training corpus

Before training on external text, prepare a source manifest and run the provenance-gated cleaner:

```bash
python -m core.prepare_corpus --input-dir data/raw --sources data/sources.json --output data/prepared/corpus.txt
```

Every input text file must have a source URL, exact license name, and explicit `training_allowed: true` entry. The script normalizes Unicode, deduplicates paragraphs, and writes a SHA-256 provenance manifest. It cannot make legal determinations; verify each license yourself. See [corpus provenance guidance](data/CORPUS_SOURCES.md).

## Train and run Core Transformer

The trainer uses the custom byte-level BPE tokenizer. The included sample corpus is synthetic and far too small to produce a useful language model; it is suitable only for a short code-path smoke test.

```bash
python -m core.transformer_train --input data/sample.txt --output artifacts/core-0.1.pt --steps 2 --batch-size 2 --context-length 16 --vocab-size 256
python -m core.transformer_generate --checkpoint artifacts/core-0.1.pt --prompt "Ai'Hub" --max-new-tokens 20
```

To evaluate on a separate held-out text file, use `core.transformer_eval`. For real experiments, supply a legally usable, sufficiently large corpus and record its origin and license in `data/CORPUS_SOURCES.md`.

## Export for mobile inference

Export a trained checkpoint to a dynamic-sequence ONNX graph and verify it against PyTorch:

```bash
python -m core.export_onnx --checkpoint artifacts/core-0.1.pt --output artifacts/core-0.1.onnx
```

The export includes logits only; the Android host must apply the same byte-level BPE tokenizer and perform autoregressive sampling by repeatedly invoking the ONNX graph. ONNX export and runtime validation do not themselves mean the Android app has integrated the model. The exported graph is not bundled into the current APK.

The canonical architecture is 4 layers, 4 heads, 256-dimensional embeddings, 1024-dimensional feed-forward layers, and a 512-token context. Training cost rises with context, batch size, and corpus size.

## Mobile app

A mobile-first installable PWA and a native Android WebView shell are included. The current APK still bundles a tiny statistical bigram demo, not the Transformer or ONNX runtime. See [Android build and install instructions](android/README.md).

## Known limitations

- No curated training corpus, pretrained weights, benchmark results, or validated model release exists.
- The ONNX graph is a deployment milestone only; native Android integration, tokenizer parity, performance profiling, and model packaging remain to be completed.
- No claim of independent reasoning capability is made at this stage.

## License

The repository's software is licensed under the MIT License. This does not grant rights to third-party datasets; each dataset must retain its own license and provenance.
