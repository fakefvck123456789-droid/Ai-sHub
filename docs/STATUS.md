# Implementation status

This repository is an active foundation build, not a finished AI product or trained model release.

## Implemented in the repository

- **Core architecture:** decoder-only Transformer with RoPE, RMSNorm, causal multi-head attention, SwiGLU, tied embeddings, and next-token loss.
- **Tokenizer:** custom byte-level BPE with deterministic ranked merges, bounded training samples, Unicode-safe round trips, and chunked encoding.
- **Training:** CPU-first training loop, held-out validation, checkpoint save/resume with optimizer and RNG state, corpus/config/tokenizer identity checks.
- **Evaluation:** separate-corpus next-token loss and perplexity CLI.
- **Inference/router:** local generation CLI, explicit capability router, and loopback HTTP server with health, model listing, generation, request limits, and restricted local PWA CORS. Unsupported tasks are rejected; there is no external fallback.
- **Corpus preparation:** explicit per-file source/license manifest, training-permission gate, Unicode normalization, paragraph deduplication, and SHA-256 provenance output.
- **App:** installable mobile-first PWA for the local model server. It is not a compiled native Android APK.
- **Automation:** GitHub Actions runs the Python test suite. The latest verified code revision passed 33 tests.

## Not yet complete

- No real-world training corpus is included. The bundled sample is synthetic and only suitable for smoke tests.
- No useful pretrained checkpoint, benchmark result, or validated Core model release exists.
- The model has not been trained or benchmarked on the target Redmi Android device.
- PyTorch installation and execution in Android/Termux have not been verified.
- There is no native Android APK yet.
- Specialized Code, Vision, Image, Voice, and Motion models are not implemented.
- Advanced training features such as learning-rate scheduling, gradient accumulation, mixed precision, and distributed training are not implemented.

## What is required before calling Core 0.1 finished

1. Add a sufficiently large corpus whose license and training permission have been independently verified.
2. Run reproducible training on supported hardware and preserve the corpus manifest and training configuration.
3. Evaluate on a genuinely separate held-out corpus and publish metrics and limitations.
4. Benchmark generation latency and memory on the intended deployment hardware.
5. Complete a security review and decide whether a native Android package is required in addition to the PWA.

Do not treat a successful smoke test, falling training loss, or generated sample as proof of general reasoning ability.
