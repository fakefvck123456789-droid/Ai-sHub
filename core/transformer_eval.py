"""Evaluate an Ai'Hub checkpoint on a separate UTF-8 text file."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from core.bpe_tokenizer import BytePairTokenizer
from core.transformer import AiHubTransformer, TransformerConfig


@torch.inference_mode()
def evaluate_checkpoint(checkpoint_path: Path, text_path: Path,
                        device: str = "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("project") != "Ai'Hub Core 0.1":
        raise ValueError("Unsupported Ai'Hub checkpoint format")
    tokenizer = BytePairTokenizer.from_dict(checkpoint["tokenizer"])
    config = TransformerConfig.from_dict(checkpoint["config"])
    model = AiHubTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Evaluation text is empty")
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        raise ValueError("Evaluation text must contain at least two tokens")

    total_nll = 0.0
    total_targets = 0
    context = config.context_length
    target_block = max(1, context // 2)
    prefix_length = context - target_block + 1
    for target_start in range(1, len(tokens), target_block):
        target_end = min(target_start + target_block, len(tokens))
        input_start = max(0, target_start - prefix_length)
        x = torch.tensor([tokens[input_start:target_end - 1]],
                         dtype=torch.long, device=device)
        y = torch.tensor([tokens[input_start + 1:target_end]],
                         dtype=torch.long, device=device)
        offset = target_start - 1 - input_start
        logits, _ = model(x)
        selected_logits = logits[:, offset:, :].contiguous()
        selected_targets = y[:, offset:].contiguous()
        count = target_end - target_start
        nll = F.cross_entropy(selected_logits.view(-1, selected_logits.size(-1)),
                              selected_targets.view(-1), reduction="sum")
        total_nll += float(nll.cpu())
        total_targets += count

    average_loss = total_nll / total_targets
    perplexity = math.exp(min(average_loss, 80.0))
    return {"evaluation_tokens": len(tokens), "target_tokens": total_targets,
            "average_loss": average_loss, "perplexity": perplexity,
            "checkpoint": str(checkpoint_path), "device": device}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="Separate UTF-8 evaluation corpus")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.input, args.device), indent=2))


if __name__ == "__main__":
    main()
