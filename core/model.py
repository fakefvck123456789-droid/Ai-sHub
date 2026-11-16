"""Transparent bigram language-model prototype; not a neural network."""
from __future__ import annotations

from collections import Counter, defaultdict
import random


class NGramModel:
    """A small next-token frequency model used to test the training pipeline."""

    def __init__(self, smoothing: float = 0.1):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = float(smoothing)
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()

    def fit(self, sequences: list[list[int]]) -> "NGramModel":
        for sequence in sequences:
            for token in sequence:
                self.unigram[str(token)] += 1
            for left, right in zip(sequence, sequence[1:]):
                self.counts[str(left)][str(right)] += 1
        return self

    def next_token(self, previous: int | None, rng: random.Random | None = None) -> int:
        rng = rng or random
        distribution = self.counts.get(str(previous), Counter()) if previous is not None else Counter()
        if not distribution:
            distribution = self.unigram
        if not distribution:
            return 0
        items = sorted((int(token), count + self.smoothing) for token, count in distribution.items())
        total = sum(weight for _, weight in items)
        point = rng.random() * total
        for token, weight in items:
            point -= weight
            if point <= 0:
                return token
        return items[-1][0]

    def generate(self, prompt: list[int], max_new_tokens: int = 30, seed: int | None = None) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        result = list(prompt)
        rng = random.Random(seed)
        for _ in range(max_new_tokens):
            previous = result[-1] if result else None
            result.append(self.next_token(previous, rng))
        return result

    def to_dict(self) -> dict:
        return {
            "type": "bigram",
            "smoothing": self.smoothing,
            "unigram": dict(self.unigram),
            "counts": {key: dict(value) for key, value in self.counts.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NGramModel":
        if payload.get("type") != "bigram":
            raise ValueError("Unsupported model type.")
        model = cls(smoothing=float(payload.get("smoothing", 0.1)))
        model.unigram.update({str(k): int(v) for k, v in payload.get("unigram", {}).items()})
        for key, values in payload.get("counts", {}).items():
            model.counts[str(key)].update({str(k): int(v) for k, v in values.items()})
        return model
