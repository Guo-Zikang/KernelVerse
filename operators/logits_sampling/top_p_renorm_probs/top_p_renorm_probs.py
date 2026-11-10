from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.top_p_renorm_probs.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Mask probabilities outside top-p and renormalize them."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.top_p = float(shape["top_p"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        probs_compute = probs.to(self.compute_dtype)
        sorted_probs, indices = torch.sort(probs_compute, dim=-1, descending=False)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask_sorted = (cdf >= 1.0 - self.top_p).to(torch.int32)
        mask = torch.zeros_like(mask_sorted)
        mask.scatter_add_(1, indices, mask_sorted)
        masked = torch.where(mask.bool(), probs_compute, torch.zeros_like(probs_compute))
        return (masked / masked.sum(dim=-1, keepdim=True)).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    vocab_size = int(shape["vocab_size"])
    probs = torch.rand((batch_size, vocab_size), dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return (probs.to(dtype),)
