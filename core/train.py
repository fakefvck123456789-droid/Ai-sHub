"""Train and save the Ai'Hub n-gram pipeline prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.model import NGramModel
from core.tokenizer import WordTokenizer


def train(input_path: Path, output_path: Path) -> dict:
    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Training input is empty: {input_path}")
    tokenizer = WordTokenizer().fit(text)
    sequences = [tokenizer.encode(line) for line in text.splitlines() if line.strip()]
    model = NGramModel().fit(sequences)
    payload = {
        "format_version": 1,
        "project": "Ai'Hub Core",
        "prototype": "bigram_pipeline_test",
        "warning": "This is a statistical n-gram baseline, not the planned Transformer.",
        "tokenizer": tokenizer.to_dict(),
        "model": model.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"tokens": sum(map(len, sequences)), "vocabulary": len(tokenizer.vocabulary), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 text corpus")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ngram.json"))
    args = parser.parse_args()
    stats = train(args.input, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
