# Ai'Hub Android application

The native Android app is designed for local inference. It loads a trained Core ONNX package when the three model assets are present; otherwise it runs the tiny bigram demonstration model.

## Build a debug APK

Requirements: JDK 17, Android SDK Platform 35, Android Build Tools 35.0.0, and Gradle 8.9 or compatible.

From this directory:

```bash
gradle assembleDebug
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`.

GitHub Actions builds the debug APK and uploads it as the `aihub-android-debug-apk` artifact.

## Export a trained Core checkpoint directly into Android assets

After training and evaluating a checkpoint, export it directly into the app's assets directory:

```bash
python -m core.export_onnx --checkpoint artifacts/core-0.1.pt --output android/app/src/main/assets/core-0.1.onnx
```

The exporter writes all three required files beside the ONNX graph:

- `core-0.1.onnx`
- `core-0.1-tokenizer.json`
- `core-0.1-model.json`

Then build the APK again. At startup the app validates the project metadata and tokenizer vocabulary, loads ONNX Runtime for Android, and uses the matching byte-level BPE tokenizer. It performs autoregressive next-token inference locally; no remote AI API is called. If model files are absent or invalid, the app logs the reason and uses the demo model instead.

The ONNX graph uses a dynamic sequence dimension and int64 token IDs. Generation currently runs one full-context forward pass per new token (no KV cache), so speed depends on the device and model size. The Android BPE implementation uses a ranked priority queue to keep tokenization practical as the merge table grows.

## Current model limitation

The repository does not yet contain a trained release checkpoint or these ONNX assets. The bundled `core-prototype.json` is a tiny bigram model trained from the synthetic sample corpus. It proves the offline Android demo path works, but it is not a useful assistant, not a Transformer, and not the planned trained Ai'Hub Core 0.1. The ONNX path is not considered device-validated until a real checkpoint is bundled and inference is tested on hardware.
