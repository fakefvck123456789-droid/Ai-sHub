# Implementation status

This repository is an active foundation build, not a finished AI product or trained model release.

## Implemented in the repository

- **Core architecture:** decoder-only Transformer with RoPE, RMSNorm, causal multi-head attention, SwiGLU, tied embeddings, and next-token loss.
- **Tokenizer:** custom byte-level BPE with deterministic ranked merges, bounded training samples, Unicode-safe round trips, and chunked encoding.
- **Training:** CPU-first training loop, held-out validation, checkpoint save/resume with optimizer and RNG state, corpus/config/tokenizer identity checks, gradient accumulation, and warmup/cosine learning-rate scheduling.
- **Evaluation/benchmarking:** separate-corpus next-token loss and perplexity CLI, plus a local generation latency/throughput and memory benchmark.
- **Inference/router:** local generation CLI, explicit capability router, and loopback HTTP server with health, model listing, generation, request limits, and restricted local PWA CORS. Unsupported tasks are rejected; there is no external fallback.
- **Corpus preparation:** explicit per-file source/license manifest, training-permission gate, Unicode normalization, paragraph deduplication, and SHA-256 provenance output.
- **App:** installable mobile-first PWA discovers selectable local models from `/models`; a native Android WebView shell and debug-APK CI build are now added. The native shell still depends on a separately running local Python inference server.
- **Automation:** GitHub Actions runs Python tests, Python bytecode compilation, and PWA JavaScript syntax checks. The latest verified code revision passed 36 tests; the new training/benchmark work is awaiting its CI run.

## Not yet complete

- No real-world training corpus is included. The bundled sample is synthetic and only suitable for smoke tests.
- No useful pretrained checkpoint, benchmark result, or validated Core model release exists.
- The model has not been trained or benchmarked on the target Redmi Android device.
- PyTorch installation and execution in Android/Termux have not been verified.
- There is no native Android APK yet.
- Specialized Code, Vision, Image, Voice, and Motion models are not implemented.
- Mixed precision and distributed training are not implemented.

## What is required before calling Core 0.1 finished

1. Add a sufficiently large corpus whose license and training permission have been independently verified.
2. Run reproducible training on supported hardware and preserve the corpus manifest and training configuration.
3. Evaluate on a genuinely separate held-out corpus and publish metrics and limitations.
4. Benchmark generation latency and memory on the intended deployment hardware.
5. Complete a security review and decide whether a native Android package is required in addition to the PWA.

Do not treat a successful smoke test, falling training loss, or generated sample as proof of general reasoning ability.
