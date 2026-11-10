from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.top_k_renorm_probs.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Mask probabilities outside top-k and renormalize them."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.top_k = int(shape["top_k"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        probs_compute = probs.to(self.compute_dtype)
        sorted_probs, _ = torch.sort(probs_compute, dim=-1, descending=True)
        pivot = sorted_probs[:, self.top_k - 1].unsqueeze(-1)
        masked = torch.where(
            probs_compute >= pivot,
            probs_compute,
            torch.zeros_like(probs_compute),
        )
        return (masked / masked.sum(dim=-1, keepdim=True)).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    vocab_size = int(shape["vocab_size"])
    probs = torch.rand((batch_size, vocab_size), dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return (probs.to(dtype),)
