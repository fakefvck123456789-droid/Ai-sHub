"""Deterministic byte-pair encoding tokenizer trained from UTF-8 text."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class BytePairTokenizer:
    """Byte-level BPE: IDs 0..255 are bytes; learned merges follow them."""
    merges: list[tuple[int, int]] = field(default_factory=list)
    min_frequency: int = 2
    _ranks: dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _expansions: list[bytes] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        self._rebuild()

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _rebuild(self):
        self._ranks = {tuple(pair): rank for rank, pair in enumerate(self.merges)}
        self._expansions = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if left >= len(self._expansions) or right >= len(self._expansions):
                raise ValueError("Merge references a token that has not been defined")
            self._expansions.append(self._expansions[left] + self._expansions[right])

    def fit(self, text: str, vocab_size: int = 512) -> "BytePairTokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")
        sequence = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        next_id = 256
        while next_id < vocab_size and len(sequence) > 1:
            counts = Counter(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < self.min_frequency:
                break
            merges.append(pair)
            merged: list[int] = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    merged.append(next_id)
                    index += 2
                else:
                    merged.append(sequence[index])
                    index += 1
            sequence = merged
            next_id += 1
        self.merges = merges
        self._rebuild()
        return self

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            merged: list[int] = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    merged.append(256 + rank)
                    index += 2
                else:
                    merged.append(sequence[index])
                    index += 1
            sequence = merged
        return sequence

    def decode(self, ids: list[int]) -> str:
        data = bytearray()
        for token_id in ids:
            if not isinstance(token_id, int) or token_id < 0 or token_id >= len(self._expansions):
                raise ValueError(f"Invalid token ID: {token_id}")
            data.extend(self._expansions[token_id])
        return data.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "byte_bpe", "min_frequency": self.min_frequency,
                "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, payload: dict) -> "BytePairTokenizer":
        if payload.get("type") != "byte_bpe" or not isinstance(payload.get("merges"), list):
            raise ValueError("Unsupported tokenizer format")
        return cls(merges=[tuple(pair) for pair in payload["merges"]],
                   min_frequency=int(payload.get("min_frequency", 2)))
