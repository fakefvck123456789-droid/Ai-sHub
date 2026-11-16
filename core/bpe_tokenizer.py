"""Deterministic byte-level BPE tokenizer trained from UTF-8 text."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq


@dataclass
class BytePairTokenizer:
    """Byte-level BPE: IDs 0..255 are bytes; learned merges follow them."""
    merges: list[tuple[int, int]] = field(default_factory=list)
    min_frequency: int = 2
    max_training_bytes: int = 100_000
    _ranks: dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _expansions: list[bytes] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if self.max_training_bytes < 1:
            raise ValueError("max_training_bytes must be positive")
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
        raw = text.encode("utf-8")
        if len(raw) > self.max_training_bytes:
            chunks = min(8, max(1, self.max_training_bytes // 2))
            chunk_size = max(1, (self.max_training_bytes - (chunks - 1)) // chunks)
            last_start = len(raw) - chunk_size
            starts = ([i * last_start // (chunks - 1) for i in range(chunks)]
                      if chunks > 1 else [0])
            sample = b"\n".join(raw[start:start + chunk_size] for start in starts)
        else:
            sample = raw
        sequence = list(sample)
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

    def _encode_bytes(self, raw: bytes) -> list[int]:
        values = list(raw)
        count = len(values)
        if count < 2 or not self.merges:
            return values

        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        alive = [True] * count
        heap = []

        def push_pair(left_index: int):
            if left_index < 0 or not alive[left_index]:
                return
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index]:
                return
            pair = (values[left_index], values[right_index])
            rank = self._ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, left_index, pair[0], pair[1]))

        for index in range(count - 1):
            push_pair(index)

        while heap:
            rank, left_index, left_token, right_token = heapq.heappop(heap)
            if not alive[left_index]:
                continue
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index]:
                continue
            if values[left_index] != left_token or values[right_index] != right_token:
                continue
            values[left_index] = 256 + rank
            alive[right_index] = False
            next_index = following[right_index]
            following[left_index] = next_index
            if next_index >= 0:
                previous[next_index] = left_index
            push_pair(previous[left_index])
            push_pair(left_index)

        output = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(values[index])
            index = following[index]
        return output

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        if not raw:
            return []
        # Bound temporary linked-list and heap allocations for large corpora.
        result = []
        chunk_bytes = 8192
        for start in range(0, len(raw), chunk_bytes):
            result.extend(self._encode_bytes(raw[start:start + chunk_bytes]))
        return result

    def decode(self, ids: list[int]) -> str:
        data = bytearray()
        for token_id in ids:
            if not isinstance(token_id, int) or token_id < 0 or token_id >= len(self._expansions):
                raise ValueError(f"Invalid token ID: {token_id}")
            data.extend(self._expansions[token_id])
        return data.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "byte_bpe", "min_frequency": self.min_frequency,
                "max_training_bytes": self.max_training_bytes,
                "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, payload: dict) -> "BytePairTokenizer":
        if payload.get("type") != "byte_bpe" or not isinstance(payload.get("merges"), list):
            raise ValueError("Unsupported tokenizer format")
        return cls(merges=[tuple(pair) for pair in payload["merges"]],
                   min_frequency=int(payload.get("min_frequency", 2)),
                   max_training_bytes=int(payload.get("max_training_bytes", 100_000)))
