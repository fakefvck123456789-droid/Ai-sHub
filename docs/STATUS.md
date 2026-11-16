# Project status

## Working in the Android prototype

- Native Android WebView shell bundles the app interface and runs offline.
- A small statistical bigram model remains bundled as the fallback demonstration asset.
- Android includes ONNX Runtime for Android and a native Java inference bridge.
- When a valid trained Core package is present, Android loads the ONNX graph and the matching byte-level BPE tokenizer and performs autoregressive inference locally.
- Android BPE encoding uses ranked merge candidates and a priority queue to avoid repeatedly scanning the full merge table.
- The app can open without Termux, Python, a remote server, API keys, or an account.
- Android debug APK builds and Python tests run in GitHub Actions.

## Important limitation

The repository does not yet contain a trained release checkpoint or a legally reviewed real-world training corpus. The bundled fallback model is a tiny statistical bigram model trained from synthetic sample text. It is not a neural network, Transformer, reasoning model, or useful general-purpose assistant.

The ONNX Android path is implemented but remains dormant until a trained checkpoint is exported and its three generated files are bundled into the APK. A successful debug build alone does not validate inference on a physical device.

## Next release gates

1. Select and document legally usable training data, including language coverage and license provenance.
2. Train Core 0.1 and evaluate it on a separate held-out corpus; record quality, memory, and latency.
3. Export the checkpoint to ONNX and package the graph, tokenizer, and metadata in Android assets.
4. Test real inference on target Android devices, including tokenizer parity, startup, memory, latency, and battery.
5. Replace the fallback demo only after the trained model passes those checks.
6. Add specialized model families only after Core is viable.
