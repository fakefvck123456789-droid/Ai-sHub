# Ai'Hub

An independent, research-first AI software and model ecosystem. Ai'Hub is designed to develop and run its own models and tooling rather than proxying prompts to third-party AI APIs.

## Current status

**Foundation stage — Ai'Hub Core 0.1 architecture and training pipeline are implemented; no trained release model exists.**

The repository contains:
- A dependency-light bigram baseline for testing the original data pipeline.
- A PyTorch decoder-only Transformer with causal attention, RoPE, RMSNorm, SwiGLU, next-token loss, and autoregressive generation.
- A custom byte-level BPE tokenizer.
- A minimal training and checkpoint inference CLI.

Neither path includes pretrained weights. A successful run on the tiny sample corpus is only a software smoke test, not evidence of useful language ability or general reasoning.

## Repository layout

- `core/` — tokenizer, bigram baseline, Transformer, training, generation, and local API server
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
python -m pip install -r requirements.txt
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

The trainer now uses the custom byte-level BPE tokenizer. The included sample corpus is synthetic and far too small to produce a useful language model; it is suitable only for a short code-path smoke test.

```bash
python -m core.transformer_train --input data/sample.txt --output artifacts/core-0.1.pt --steps 2 --batch-size 2 --context-length 16 --vocab-size 256
python -m core.transformer_generate --checkpoint artifacts/core-0.1.pt --prompt "Ai'Hub" --max-new-tokens 20
python -m core.transformer_server --checkpoint artifacts/core-0.1.pt --host 127.0.0.1 --port 8766
```

To evaluate on a separate held-out text file (create `data/heldout.txt` from data not used for training):

```bash
python -m core.transformer_eval --checkpoint artifacts/core-0.1.pt --input data/heldout.txt
```

This reports average next-token loss and perplexity. Compare runs only when they use the same evaluation corpus and tokenizer; a tiny or overlapping corpus is not a meaningful benchmark.

The canonical architecture is 4 layers, 4 heads, 256-dimensional embeddings, 1024-dimensional feed-forward layers, and a 512-token context. The training CLI allows shorter contexts for smoke tests and constrained experiments. Training cost rises quickly with context, batch size, and corpus size.

For real experiments, supply a legally usable, sufficiently large corpus and record its origin and license in `data/CORPUS_SOURCES.md`. Train and resume against the exact same corpus file:

```bash
python -m core.transformer_train --input data/prepared/corpus.txt --output artifacts/core-0.1.pt --steps 1000 --context-length 128
python -m core.transformer_train --input data/prepared/corpus.txt --output artifacts/core-0.1.pt --steps 2000 --context-length 128 --resume artifacts/core-0.1.pt
```

Here `--steps` is the target total step count, not additional steps. Resume verifies the corpus hash, tokenizer, model configuration, and validation split. See [Transformer implementation notes](docs/TRANSFORMER.md).


## Mobile app (PWA)

A mobile-first installable web app is included in `app/`. It is a PWA, not a compiled native APK. It communicates with the local Transformer server only; no external AI API is used.

With the Transformer server running in one terminal, serve the app in another:

```bash
python -m http.server 8080 --bind 127.0.0.1 --directory app
```

Open `http://127.0.0.1:8080` on the same device. On Android, use the browser's **Add to Home screen** / **Install app** option if available. See [app setup](app/README.md). The API restricts browser CORS to the two local origins on port 8080.

## Android / Termux

The bigram prototype can run with standard Python. PyTorch installation support varies by Android device, Python version, and available package builds; `pip install -r requirements.txt` is not guaranteed to work in Termux. For Transformer training, use a supported Linux, Windows, or macOS CPU environment unless a compatible Android PyTorch build has been independently verified.

## Known limitations

- The Transformer trainer is a minimal single-corpus loop with a held-out validation slice and validation loss reporting. Learning-rate scheduling, mixed precision, gradient accumulation, and distributed training are not implemented yet. Checkpoint resume stores optimizer and RNG state and verifies corpus identity.
- Byte-level BPE is a simple reference implementation, not optimized for large corpora.
- No curated training corpus, pretrained weights, benchmark results, or validated model release exists.
- No claim of independent reasoning capability is made at this stage.

## License

The repository's software is licensed under the MIT License. This does not grant rights to third-party datasets; each dataset must retain its own license and provenance.
