"""Export Ai'Hub Core checkpoints and their tokenizer for Android ONNX Runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from core.bpe_tokenizer import BytePairTokenizer
from core.transformer import AiHubTransformer, TransformerConfig


class LogitsOnly(nn.Module):
    def __init__(self, model: AiHubTransformer):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(token_ids)
        return logits


def export_checkpoint(checkpoint_path: Path, output_path: Path,
                      opset: int = 17, verify: bool = True) -> dict:
    if opset < 17:
        raise ValueError("ONNX opset 17 or newer is required")
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install ONNX export dependencies: pip install onnx onnxruntime") from error

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("project") != "Ai'Hub Core 0.1":
        raise ValueError("Unsupported Ai'Hub checkpoint format")
    config = TransformerConfig.from_dict(payload["config"])
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint tokenizer vocabulary does not match model configuration")
    model = AiHubTransformer(config).cpu()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    wrapper = LogitsOnly(model).eval()

    sample_length = min(8, config.context_length)
    sample = torch.randint(0, config.vocab_size, (1, sample_length), dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (sample,), str(output_path),
        input_names=["token_ids"], output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch", 1: "sequence"},
                      "logits": {0: "batch", 1: "sequence"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    import onnx
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)

    tokenizer_path = output_path.with_name("core-0.1-tokenizer.json")
    metadata_path = output_path.with_name("core-0.1-model.json")
    tokenizer_path.write_text(json.dumps(tokenizer.to_dict(), ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps({
        "project": "Ai'Hub Core 0.1",
        "config": config.__dict__,
        "tokenizer_file": tokenizer_path.name,
        "model_file": output_path.name,
    }, indent=2), encoding="utf-8")

    max_absolute_error = None
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("ONNX Runtime is required for export verification") from error
        import numpy as np
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        max_absolute_error = 0.0
        for length in sorted({1, sample_length, min(config.context_length, sample_length + 3)}):
            ids = torch.randint(0, config.vocab_size, (1, length), dtype=torch.long)
            with torch.inference_mode():
                expected = wrapper(ids).cpu().numpy()
            actual = session.run(["logits"], {"token_ids": ids.numpy()})[0]
            max_absolute_error = max(max_absolute_error, float(np.max(np.abs(expected - actual))))
        if max_absolute_error > 1e-4:
            raise RuntimeError(f"ONNX parity check failed: max absolute error {max_absolute_error}")
    return {"output": str(output_path), "tokenizer": str(tokenizer_path),
            "metadata": str(metadata_path), "opset": opset,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "vocabulary": config.vocab_size, "context_length": config.context_length,
            "max_absolute_error": max_absolute_error, "verified": bool(verify)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/core-0.1.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-runtime-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output,
                                       args.opset, not args.skip_runtime_check), indent=2))


if __name__ == "__main__":
    main()
