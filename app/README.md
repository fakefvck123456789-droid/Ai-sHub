# Ai'Hub mobile web app

This is an installable, mobile-first Progressive Web App (PWA), not a compiled native Android APK. It talks to the local Transformer HTTP server; it does not call a cloud AI API.

## Run on Android with Termux

1. Install Python and clone the repository in Termux.
2. Install a compatible PyTorch build. Android/Termux support is not guaranteed; if it cannot be installed, use a supported Linux/Windows/macOS CPU machine instead.
3. Create a smoke-test checkpoint (the bundled synthetic text is not a useful training dataset):

   ```bash
   python -m core.transformer_train --input data/sample.txt --output artifacts/core-0.1.pt --steps 2 --batch-size 2 --context-length 16 --vocab-size 256
   ```

4. In one Termux session, start the local model server:

   ```bash
   python -m core.transformer_server --checkpoint artifacts/core-0.1.pt --host 127.0.0.1 --port 8766
   ```

5. In a second Termux session, from the repository root, serve the app:

   ```bash
   python -m http.server 8080 --bind 127.0.0.1 --directory app
   ```

6. Open `http://127.0.0.1:8080` in the Android browser. Use the browser menu's **Add to Home screen** / **Install app** action if available.

The app loads the available local text-generation models from the server's `/models` endpoint and includes the selected model ID in generation requests. The registry explicitly reports that external fallback is disabled.

The API allows browser requests only from `http://127.0.0.1:8080` and `http://localhost:8080` by default. Keep both servers bound to loopback. Do not bind the inference API to a public interface without authentication and a security review.

## Local computer

Use the same two commands in separate terminals, then open the local app URL in a browser.

The app shell can be cached for offline loading, but generation still requires the local Python model server and checkpoint to be running.
