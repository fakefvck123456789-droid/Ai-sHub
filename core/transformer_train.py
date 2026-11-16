"""Train Ai'Hub Core Transformer from a UTF-8 text corpus (experimental)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from core.bpe_tokenizer import BytePairTokenizer
from core.transformer import AiHubTransformer, TransformerConfig


def make_batches(tokens: list[int], context: int, batch_size: int, device: str):
    if len(tokens) < context + 1:
        raise ValueError(f"Corpus needs at least {context + 1} tokens; got {len(tokens)}")
    data = torch.tensor(tokens, dtype=torch.long)
    max_start = len(data) - context - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x.to(device), y.to(device)


def train(corpus_path: Path, output_path: Path, steps: int = 1000,
          batch_size: int = 8, learning_rate: float = 3e-4,
          context_length: int = 128, device: str = "cpu", seed: int = 42,
          vocab_size: int = 512) -> dict:
    if steps < 1 or batch_size < 1 or context_length < 1 or learning_rate <= 0:
        raise ValueError("steps, batch_size, context_length and learning_rate must be positive")
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 for byte-level BPE")
    random.seed(seed)
    torch.manual_seed(seed)
    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Training corpus is empty")
    tokenizer = BytePairTokenizer().fit(text, vocab_size=vocab_size)
    tokens = tokenizer.encode(text)
    if len(tokens) < context_length + 1:
        raise ValueError("Corpus is too short for the selected context length")
    config = TransformerConfig(vocab_size=tokenizer.vocab_size,
                               context_length=context_length)
    model = AiHubTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    last_loss = None
    for _ in range(steps):
        x, y = make_batches(tokens, context_length, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "project": "Ai'Hub Core 0.1",
        "status": "experimental_unvalidated_training_run",
        "config": model.config_dict(),
        "tokenizer": tokenizer.to_dict(),
        "model_state_dict": model.state_dict(),
        "training": {"steps": steps, "batch_size": batch_size,
                     "learning_rate": learning_rate, "context_length": context_length,
                     "seed": seed, "final_training_loss": last_loss},
    }, output_path)
    return {"steps": steps, "tokens": len(tokens), "vocabulary": tokenizer.vocab_size,
            "learned_merges": len(tokenizer.merges), "final_training_loss": last_loss,
            "output": str(output_path), "device": device,
            "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/core-0.1.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = train(args.input, args.output, args.steps, args.batch_size,
                  args.learning_rate, args.context_length, args.device, args.seed,
                  args.vocab_size)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
