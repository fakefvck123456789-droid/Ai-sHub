# Project status

## Working in the Android prototype

- Native Android WebView shell bundles the app interface.
- A small statistical bigram model is bundled as an app asset.
- Android loads the model into an in-process Java runtime at startup.
- Prompt tokenization, next-token sampling, and decoding happen locally on-device.
- The app can open and generate text without Termux, Python, a server, API keys, accounts, or internet access.
- Android debug APK build is automated in GitHub Actions.

## Important limitation

The bundled model is a pipeline smoke test trained from a tiny synthetic corpus. It is not a neural network, Transformer, reasoning model, or useful general-purpose assistant. Do not describe this prototype as equivalent to a production AI assistant.

## Remaining work

- Implement and train Ai'Hub Core 0.1 Transformer on a legally usable, documented corpus.
- Export trained weights to an Android-compatible runtime and replace the bigram demo.
- Benchmark memory, latency, battery use, and generation quality on target Android devices.
- Add specialized model families only after Core is viable.
