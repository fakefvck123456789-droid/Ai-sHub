"""Measure local checkpoint generation latency and throughput."""
from __future__ import annotations

import argparse
import json
import platform
try:
    import resource
except ImportError:  # Windows
    resource = None
import statistics
import time
from pathlib import Path

import torch

from core.transformer_generate import load_checkpoint


def benchmark(checkpoint: Path, prompt: str, new_tokens: int = 32,
              warmup: int = 1, repeats: int = 3, device: str = "cpu") -> dict:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if new_tokens < 1 or warmup < 0 or repeats < 1:
        raise ValueError("new_tokens and repeats must be positive; warmup cannot be negative")
    model, tokenizer = load_checkpoint(checkpoint, device)
    encoded = tokenizer.encode(prompt)
    if not encoded:
        raise ValueError("prompt must contain at least one encodable byte")
    ids = torch.tensor([encoded], dtype=torch.long, device=device)

    def run_once():
        with torch.inference_mode():
            return model.generate(ids.clone(), max_new_tokens=new_tokens,
                                  temperature=0.8, top_k=40)

    for _ in range(warmup):
        run_once()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_once()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)

    peak_rss_mb = None
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = usage / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    result = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "prompt_tokens": len(encoded),
        "generated_tokens_per_run": new_tokens,
        "repeats": repeats,
        "warmup_runs": warmup,
        "latency_seconds_mean": statistics.mean(durations),
        "latency_seconds_median": statistics.median(durations),
        "tokens_per_second_mean": new_tokens / statistics.mean(durations),
        "process_peak_rss_mb": peak_rss_mb,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "warning": "Benchmark is hardware- and load-dependent; process peak RSS includes Python and model loading.",
    }
    if str(device).startswith("cuda"):
        result["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Ai'Hub")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.checkpoint, args.prompt, args.new_tokens,
                               args.warmup, args.repeats, args.device), indent=2))


if __name__ == "__main__":
    main()
