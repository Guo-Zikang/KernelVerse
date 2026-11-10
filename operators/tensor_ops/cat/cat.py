from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.cat.helpers import (
    _normalize_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Concatenate tensors using torch.cat."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        input_shape = tuple(int(dim) for dim in shape["input_shape"])
        self.dim = _normalize_dim(shape["dim"], input_shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors_compute = [tensor.to(self.compute_dtype) for tensor in tensors]
        output = torch.cat(tensors_compute, dim=self.dim)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    input_shape = tuple(int(dim) for dim in shape["input_shape"])
    num_tensors = int(shape["num_tensors"])
    dtype = _resolve_dtype(shape["dtype"])

    tensors = tuple(
        torch.randn(input_shape, dtype=torch.float32).to(dtype)
        for _ in range(num_tensors)
    )
    return tensors
