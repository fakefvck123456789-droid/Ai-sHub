# Ai'Hub Android application

The Android app uses AndroidX WebViewAssetLoader to serve the bundled interface from a local HTTPS app-assets origin. It does not bind a localhost port or require internet access. Inference is intended to run locally.

## Build a debug APK

Requirements: JDK 17, Android SDK Platform 35, Android Build Tools 35.0.0, and Gradle 8.9.

From this directory:

    gradle assembleDebug

APK output: app/build/outputs/apk/debug/app-debug.apk.

GitHub Actions builds the debug APK and uploads it as the aihub-android-debug-apk artifact.

## Core ONNX package

After training and evaluating a checkpoint, export it directly into the Android assets directory:

    python -m core.export_onnx --checkpoint artifacts/core-0.1.pt --output android/app/src/main/assets/core-0.1.onnx

The exporter writes three files:

- core-0.1.onnx
- core-0.1-tokenizer.json
- core-0.1-model.json

Build the APK again after exporting. At startup, the app validates metadata and tokenizer vocabulary, then loads ONNX Runtime for Android. Generation is autoregressive and local. If the model package is absent or invalid, the app falls back to the statistical demo and records a startup warning instead of terminating during model initialization.

The ONNX graph uses dynamic sequence length and int64 token IDs. Generation currently runs one full-context forward pass per token (no KV cache), so performance depends on model size and device hardware.

## Current limitation

No trained bilingual Core checkpoint or ONNX model package is currently bundled. The included core-prototype.json is a tiny bigram demo trained on synthetic text. It is not the planned Transformer and is not a useful general assistant.

CI verifies compilation, not crash-free behavior on every Android device. Physical-device tests are still required for cold start, repeated generations, rotation, background/foreground transitions, low-memory conditions, and offline use. If the app still closes, capture Android logcat around the crash; the exception trace is needed to identify device-specific failures.
