"""Composition helpers for benchmark, backend, and model prompt layers."""

from __future__ import annotations


def compose_generation_prompt(
    benchmark_prompt: str,
    backend_prompt: str,
    model_adapter_prompt: str,
) -> str:
    """Compose prompt layers in benchmark, backend, then model order."""
    layers = [benchmark_prompt.rstrip(), backend_prompt.strip()]
    normalized_adapter = model_adapter_prompt.strip()
    if normalized_adapter:
        layers.append(normalized_adapter)
    return "\n\n".join(layers) + "\n"
