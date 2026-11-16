"""Local HTTP inference server for an Ai'Hub Core Transformer checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from core.bpe_tokenizer import BytePairTokenizer
from core.transformer import AiHubTransformer, TransformerConfig


def load_checkpoint(path: Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1 or payload.get("project") != "Ai'Hub Core 0.1":
        raise ValueError("Unsupported Ai'Hub checkpoint format")
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    config = TransformerConfig.from_dict(payload["config"])
    model = AiHubTransformer(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return tokenizer, model


def make_handler(tokenizer: BytePairTokenizer, model: AiHubTransformer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {
                    "status": "ok",
                    "model": "Ai'Hub Core 0.1 Transformer",
                    "checkpoint_loaded": True,
                    "vocabulary_size": tokenizer.vocab_size,
                    "context_length": model.config.context_length,
                })
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": "not_found"})
                return
            try:
                raw_length = self.headers.get("Content-Length", "0")
                length = int(raw_length)
                if length < 0:
                    raise ValueError("invalid Content-Length")
                if length > 16_384:
                    self._send(413, {"error": "request_too_large"})
                    return
                request = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                prompt = request.get("prompt", "")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("prompt must be a non-empty string")
                max_new = int(request.get("max_new_tokens", 64))
                temperature = float(request.get("temperature", 0.8))
                top_k_value = request.get("top_k", 40)
                top_k = None if top_k_value is None else int(top_k_value)
                if not 0 <= max_new <= 256:
                    raise ValueError("max_new_tokens must be between 0 and 256")
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("temperature must be a finite positive number")
                if top_k is not None and top_k < 1:
                    raise ValueError("top_k must be positive or null")
                input_ids = tokenizer.encode(prompt)
                if len(input_ids) > model.config.context_length:
                    input_ids = input_ids[-model.config.context_length:]
                tokens = torch.tensor([input_ids], dtype=torch.long,
                                      device=next(model.parameters()).device)
                with torch.inference_mode():
                    output = model.generate(tokens, max_new_tokens=max_new,
                                            temperature=temperature, top_k=top_k)
                self._send(200, {
                    "text": tokenizer.decode(output[0].tolist()),
                    "model": "Ai'Hub Core 0.1 Transformer",
                    "warning": "Experimental checkpoint; output quality is not validated.",
                })
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": "invalid_request", "detail": str(exc)})

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    tokenizer, model = load_checkpoint(args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tokenizer, model))
    print(f"Ai'Hub Transformer server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
