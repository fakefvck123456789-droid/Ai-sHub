"""Local HTTP inference server for a trained n-gram artifact."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.model import NGramModel
from core.tokenizer import WordTokenizer


def load_artifact(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = WordTokenizer.from_dict(payload["tokenizer"])
    model = NGramModel.from_dict(payload["model"])
    return tokenizer, model


def make_handler(tokenizer, model):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": "Ai'Hub Core bigram prototype"})
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 16_384:
                    self._send(413, {"error": "request_too_large"})
                    return
                request = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(request.get("prompt", ""))
                max_new = int(request.get("max_new_tokens", 30))
                if not 0 <= max_new <= 200:
                    raise ValueError("max_new_tokens must be between 0 and 200")
                generated = model.generate(tokenizer.encode(prompt), max_new, request.get("seed"))
                self._send(200, {
                    "text": tokenizer.decode(generated),
                    "prototype": "bigram_pipeline_test",
                    "warning": "Statistical demo only; not a general-purpose AI model.",
                })
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": "invalid_request", "detail": str(exc)})

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    tokenizer, model = load_artifact(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tokenizer, model))
    print(f"Ai'Hub prototype listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
