# Ai'Hub Android application

The native Android app runs inference on-device. It first looks for the trained Core ONNX assets; if they are absent, it falls back to the tiny bigram demonstration model.

## Build a debug APK

Requirements: JDK 17, Android SDK Platform 35, Android Build Tools 35.0.0, and Gradle 8.9 or compatible.

From this directory:

```bash
gradle assembleDebug
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`.

GitHub Actions builds the debug APK and uploads it as the `aihub-android-debug-apk` artifact.

## Bundle a trained Core checkpoint

After training a checkpoint and exporting it:

```bash
python -m core.export_onnx --checkpoint artifacts/core-0.1.pt --output artifacts/core-0.1.onnx
```

Copy all three generated files into `android/app/src/main/assets/`:

- `core-0.1.onnx`
- `core-0.1-tokenizer.json`
- `core-0.1-model.json`

Then build the APK again. At startup the app loads ONNX Runtime for Android and uses the exported byte-level BPE tokenizer. It performs autoregressive next-token inference locally; no remote AI API is called. If the model files are absent or invalid, it logs the reason and uses the demo model instead.

The ONNX graph uses a dynamic sequence dimension and int64 token IDs. Generation is currently one full-context forward pass per new token (no KV cache), so speed depends on the device and model size.

## Current model limitation

The repository does not yet contain a trained release checkpoint or these ONNX assets. The bundled `core-prototype.json` is a tiny bigram model trained from the synthetic sample corpus. It proves the offline Android inference path works, but it is not a useful assistant, not a Transformer, and not the planned trained Ai'Hub Core 0.1.
