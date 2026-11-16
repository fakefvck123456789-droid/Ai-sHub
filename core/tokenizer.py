"""A tiny deterministic whitespace tokenizer for the n-gram pipeline test.

This is deliberately not the planned BPE tokenizer. It keeps the first
prototype dependency-free and makes the later tokenizer replacement explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class WordTokenizer:
    """Vocabulary tokenizer with stable IDs and explicit unknown handling."""

    vocabulary: list[str] = field(default_factory=list)
    _to_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def fit(self, text: str) -> "WordTokenizer":
        tokens = TOKEN_RE.findall(text)
        self.vocabulary = ["<unk>"] + sorted(set(tokens) - {"<unk>"})
        self._to_id = {token: index for index, token in enumerate(self.vocabulary)}
        return self

    def encode(self, text: str) -> list[int]:
        if not self._to_id:
            raise ValueError("Tokenizer has no vocabulary; call fit() first.")
        return [self._to_id.get(token, 0) for token in TOKEN_RE.findall(text)]

    def decode(self, ids: list[int]) -> str:
        if not self.vocabulary:
            raise ValueError("Tokenizer has no vocabulary.")
        tokens = [self.vocabulary[i] if 0 <= i < len(self.vocabulary) else "<unk>" for i in ids]
        output = ""
        for token in tokens:
            if token in ".,!?;:)]}" or token == "'":
                output = output.rstrip() + token
            elif token in "([{":
                output += token
            else:
                output += (" " if output else "") + token
        return output

    def to_dict(self) -> dict:
        return {"type": "word", "vocabulary": self.vocabulary}

    @classmethod
    def from_dict(cls, payload: dict) -> "WordTokenizer":
        if payload.get("type") != "word" or not isinstance(payload.get("vocabulary"), list):
            raise ValueError("Unsupported tokenizer format.")
        instance = cls(vocabulary=list(payload["vocabulary"]))
        instance._to_id = {token: index for index, token in enumerate(instance.vocabulary)}
        return instance
