"""Generate text from an Ai'Hub Core Transformer checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.bpe_tokenizer import BytePairTokenizer
from core.transformer import AiHubTransformer, TransformerConfig


def load_checkpoint(path: Path, device: str = "cpu"):
    # Checkpoints are created by this project and contain only tensors and basic data.
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1 or payload.get("project") != "Ai'Hub Core 0.1":
        raise ValueError("Unsupported Ai'Hub checkpoint format")
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    config = TransformerConfig.from_dict(payload["config"])
    model = AiHubTransformer(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer


def generate(checkpoint: Path, prompt: str, max_new_tokens: int = 80,
             temperature: float = 0.8, top_k: int = 40, device: str = "cpu") -> str:
    model, tokenizer = load_checkpoint(checkpoint, device)
    encoded = tokenizer.encode(prompt)
    if not encoded:
        raise ValueError("Prompt must contain at least one UTF-8 byte")
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k)
    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(generate(args.checkpoint, args.prompt, args.max_new_tokens,
                   args.temperature, args.top_k, args.device))


if __name__ == "__main__":
    main()
