# Ai'Hub Android application

The Android app bundles its interface and a tiny statistical Core prototype. Text generation runs in-process on the device; it does not start a Python server or call an external AI API.

## Build a debug APK

Requirements: JDK 17, Android SDK Platform 35, Android Build Tools 35.0.0, and Gradle 8.9 or compatible.

From this directory:

```bash
gradle assembleDebug
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`.

GitHub Actions builds the debug APK and uploads it as the `aihub-android-debug-apk` artifact.

## Current model limitation

The bundled `core-prototype.json` is a tiny bigram model trained from the synthetic sample corpus. It proves the offline Android inference path works, but it is not a useful assistant, not a Transformer, and not the planned Ai'Hub Core 0.1. The next model milestone is training and exporting the custom Transformer to an Android-compatible runtime.

No API key, user account, subscription, or network connection is needed for this prototype.
