from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.normalization.mean_variance_norm.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply mean-variance normalization using stored mean and inverse std."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        input_size = int(shape["shape"][-1])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )
        self.global_mean = nn.Parameter(
            torch.zeros(input_size, dtype=torch.float32).to(self.output_dtype),
            requires_grad=False,
        )
        self.global_invstd = nn.Parameter(
            torch.ones(input_size, dtype=torch.float32).to(self.output_dtype),
            requires_grad=False,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_compute = input_.to(self.compute_dtype)
        mean = self.global_mean.to(self.compute_dtype)
        invstd = self.global_invstd.to(self.compute_dtype)
        return ((input_compute - mean) * invstd).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    input_ = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (input_,)
