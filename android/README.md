# Ai'Hub Android application

This is a native Android WebView shell that bundles the PWA assets and serves them on loopback. It is not an AI model runtime: the Python Core inference server must still be running locally.

## Build a debug APK

Requirements: JDK 17, Android SDK Platform 35, Android Build Tools 35.0.0, and Gradle 8.9 or compatible.

From this directory:

```bash
gradle assembleDebug
```

APK output:

`app/build/outputs/apk/debug/app-debug.apk`

The GitHub Actions Android workflow builds this APK and uploads it as an artifact on pushes that touch Android/app files.

## Run with the local Core model

1. Install the APK on the Android device.
2. In Termux, start the Core server bound only to loopback:

   ```bash
   python -m core.transformer_server --checkpoint artifacts/core-0.1.pt --host 127.0.0.1 --port 8766
   ```

3. Open Ai'Hub. The native shell serves its bundled interface itself; no separate `python -m http.server` process is needed.
4. The app connects to `http://127.0.0.1:8766`.

You must already have a compatible Python/PyTorch installation and a trained checkpoint. The included synthetic sample can create only a smoke-test checkpoint, not a useful model.

## Network boundary

The app's embedded asset server binds to `127.0.0.1:8080`. The Android network security policy permits cleartext HTTP only to loopback hostnames. The inference API remains loopback-only. Do not expose either server to a LAN or public network.
