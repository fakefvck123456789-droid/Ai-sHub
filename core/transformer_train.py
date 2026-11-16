"""Train Ai'Hub Core Transformer from a UTF-8 text corpus (experimental)."""
from __future__ import annotations

import argparse
import hashlib
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


@torch.no_grad()
def evaluate(model: AiHubTransformer, tokens: list[int], context: int,
             batch_size: int, batches: int, device: str) -> float:
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            x, y = make_batches(tokens, context, batch_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.cpu()))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def train(corpus_path: Path, output_path: Path, steps: int = 1000,
          batch_size: int = 8, learning_rate: float = 3e-4,
          context_length: int = 128, device: str = "cpu", seed: int = 42,
          vocab_size: int = 512, eval_interval: int = 100,
          eval_batches: int = 10, validation_fraction: float = 0.1,
          resume_from: Path | None = None, bpe_training_bytes: int = 100_000) -> dict:
    if steps < 1 or batch_size < 1 or context_length < 1 or learning_rate <= 0:
        raise ValueError("steps, batch_size, context_length and learning_rate must be positive")
    if vocab_size < 256 or eval_interval < 1 or eval_batches < 1:
        raise ValueError("vocab_size must be >=256; evaluation settings must be positive")
    if bpe_training_bytes < 1:
        raise ValueError("bpe_training_bytes must be positive")
    if not 0.01 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.01 and 0.5")
    random.seed(seed)
    torch.manual_seed(seed)
    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Training corpus is empty")
    corpus_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    split_at = int(len(text) * (1.0 - validation_fraction))
    training_text, validation_text = text[:split_at], text[split_at:]
    tokenizer = BytePairTokenizer(max_training_bytes=bpe_training_bytes).fit(training_text, vocab_size=vocab_size)
    train_tokens = tokenizer.encode(training_text)
    validation_tokens = tokenizer.encode(validation_text)
    tokens = train_tokens + validation_tokens
    if len(train_tokens) < context_length + 1 or len(validation_tokens) < context_length + 1:
        raise ValueError("Corpus is too short for the selected context and validation split")
    config = TransformerConfig(vocab_size=tokenizer.vocab_size,
                               context_length=context_length)
    model = AiHubTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("project") != "Ai'Hub Core 0.1":
            raise ValueError("Resume file is not a supported Ai'Hub checkpoint")
        if checkpoint.get("corpus_sha256") != corpus_sha256:
            raise ValueError("Resume checkpoint was created from a different corpus")
        if checkpoint.get("config") != model.config_dict():
            raise ValueError("Resume checkpoint model configuration does not match this run")
        if checkpoint.get("tokenizer") != tokenizer.to_dict():
            raise ValueError("Resume checkpoint tokenizer does not match this corpus")
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume checkpoint does not contain optimizer state")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if checkpoint.get("training", {}).get("validation_fraction") != validation_fraction:
            raise ValueError("Resume checkpoint validation split does not match this run")
        start_step = int(checkpoint.get("training", {}).get("completed_steps", 0))
        if steps <= start_step:
            raise ValueError("steps must be greater than the checkpoint's completed_steps")
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        if device.startswith("cuda") and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
    model.train()
    last_train_loss = None
    validation_loss = None
    for step in range(start_step + 1, steps + 1):
        x, y = make_batches(train_tokens, context_length, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_loss = float(loss.detach().cpu())
        if step % eval_interval == 0 or step == steps:
            validation_loss = evaluate(model, validation_tokens, context_length,
                                       batch_size, eval_batches, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "project": "Ai'Hub Core 0.1",
        "status": "experimental_unvalidated_training_run",
        "corpus_sha256": corpus_sha256,
        "config": model.config_dict(),
        "tokenizer": tokenizer.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
        "training": {"steps": steps, "completed_steps": steps, "batch_size": batch_size,
                     "learning_rate": learning_rate, "context_length": context_length,
                     "seed": seed, "final_training_loss": last_train_loss,
                     "validation_loss": validation_loss,
                     "validation_fraction": validation_fraction},
    }, output_path)
    return {"steps": steps, "tokens": len(tokens), "training_tokens": len(train_tokens),
            "validation_tokens": len(validation_tokens), "vocabulary": tokenizer.vocab_size,
            "learned_merges": len(tokenizer.merges), "final_training_loss": last_train_loss,
            "validation_loss": validation_loss, "output": str(output_path),
            "device": device, "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/core-0.1.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--bpe-training-bytes", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--resume", type=Path, help="Resume from a compatible Ai'Hub checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = train(args.input, args.output, args.steps, args.batch_size,
                  args.learning_rate, args.context_length, args.device, args.seed,
                  args.vocab_size, args.eval_interval, args.eval_batches,
                  args.validation_fraction, args.resume, args.bpe_training_bytes)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
