from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.topk.helpers import _normalize_dim, _resolve_dtype, validate_shape


class Model(nn.Module):
    """Select top-k values and indices with deterministic tie-breaking."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.k = int(shape["k"])
        self.dim = _normalize_dim(shape["dim"])
        if self.dim < 0:
            self.dim += len(shape["shape"])
        self.largest = bool(shape.get("largest", True))
        self.sorted = bool(shape.get("sorted", True))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_compute = tensor.to(self.compute_dtype)
        sorted_indices = torch.argsort(
            tensor_compute,
            dim=self.dim,
            descending=self.largest,
            stable=True,
        )
        indices = sorted_indices.narrow(self.dim, 0, self.k)
        values = torch.gather(tensor_compute, self.dim, indices)
        return values.to(self.output_dtype), indices


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (tensor,)
