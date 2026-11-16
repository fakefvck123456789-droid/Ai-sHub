"""Explicit local-only capability router for Ai'Hub models."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from core.bpe_tokenizer import BytePairTokenizer
from core.transformer import AiHubTransformer


class UnsupportedTask(Exception):
    """Raised when no local model implements the requested task."""


class ModelNotFound(Exception):
    """Raised when a requested local model ID is not loaded."""


@dataclass(frozen=True)
class LocalModel:
    model_id: str
    family: str
    capabilities: tuple[str, ...]
    tokenizer: BytePairTokenizer
    model: AiHubTransformer

    def public_info(self) -> dict:
        return {
            "id": self.model_id,
            "family": self.family,
            "capabilities": list(self.capabilities),
            "status": "experimental",
            "checkpoint_loaded": True,
            "vocabulary_size": self.tokenizer.vocab_size,
            "context_length": self.model.config.context_length,
        }


class LocalModelRouter:
    """Routes only to loaded local checkpoints; it never calls external APIs."""
    def __init__(self, tokenizer: BytePairTokenizer, model: AiHubTransformer):
        self._models = {
            "core-0.1": LocalModel(
                model_id="core-0.1",
                family="core",
                capabilities=("text_generation",),
                tokenizer=tokenizer,
                model=model,
            )
        }

    @property
    def external_fallback(self) -> bool:
        return False

    def list_models(self) -> list[dict]:
        return [model.public_info() for model in self._models.values()]

    def resolve(self, task: str, model_id: str | None = None) -> LocalModel:
        if model_id is not None:
            selected = self._models.get(model_id)
            if selected is None:
                raise ModelNotFound(f"Local model is not loaded: {model_id}")
            if task not in selected.capabilities:
                raise UnsupportedTask(f"Model {model_id} does not support task: {task}")
            return selected
        for candidate in self._models.values():
            if task in candidate.capabilities:
                return candidate
        raise UnsupportedTask(f"No local model supports task: {task}")

    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int | None = 40,
                 task: str = "text_generation", model_id: str | None = None) -> dict:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        selected = self.resolve(task, model_id)
        token_ids = selected.tokenizer.encode(prompt)
        token_ids = token_ids[-selected.model.config.context_length:]
        input_tensor = torch.tensor(
            [token_ids], dtype=torch.long,
            device=next(selected.model.parameters()).device,
        )
        with torch.inference_mode():
            output = selected.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        return {
            "text": selected.tokenizer.decode(output[0].tolist()),
            "model": selected.model_id,
            "warning": "Experimental checkpoint; output quality is not validated.",
        }
