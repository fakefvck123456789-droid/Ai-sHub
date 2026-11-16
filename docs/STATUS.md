# Project status

## Android reliability work
- Replaced the custom loopback HTTP asset server with AndroidX WebViewAssetLoader and a stable HTTPS app-assets origin.
- Removed the app's unnecessary INTERNET permission and custom network-security exception.
- Demo-model load failures now enter a minimal fallback state instead of immediately closing the activity.
- ONNX runtime initialization failures are logged and fall back to the demo path.
- The JavaScript bridge reports startup warnings and generation errors.
- CI compiles the APK, but crash-free behavior is not confirmed until physical-device tests and logcat review.

## Core 0.1 target
- Languages: Bangla (bn) and English (en), one shared model and byte-level BPE tokenizer.
- Architecture: 4-layer decoder-only Transformer, 4 heads, 256 embedding dimensions, 1024 feed-forward dimensions, RoPE, RMSNorm, SwiGLU, 512-token context.
- Tokenizer target: 8,192 tokens.
- A bilingual corpus validator rejects data lacking at least 10% Bangla letters and 10% basic Latin letters.

## Still not complete
- No reviewed bilingual training corpus is in the repository.
- No trained release checkpoint, evaluated model card, or bundled Core ONNX asset exists.
- Physical-device crash, latency, memory, and battery testing remains outstanding.
- A successful CI build is not a Play Store release.

## Release gates
1. Verify data rights and source-level attribution for both languages.
2. Build a balanced corpus and separate held-out Bangla and English evaluation sets.
3. Train Core 0.1 and evaluate each language independently.
4. Export ONNX and verify numerical parity.
5. Bundle graph, tokenizer, and metadata in Android assets.
6. Test install, cold start, repeated generations, long prompts, rotation, background/foreground, low-memory behavior, and offline operation on physical devices.
7. Publish only after crash and quality gates pass.
