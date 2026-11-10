from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.take_along_dim.helpers import (
    _normalize_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Take values along one dimension using torch.take_along_dim."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.dim = _normalize_dim(shape["dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        output = torch.take_along_dim(tensor_compute, indices.long(), dim=self.dim)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    input_shape = tuple(int(dim) for dim in shape["input_shape"])
    dim = _normalize_dim(shape["dim"])
    if dim < 0:
        dim += len(input_shape)
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    indices = torch.randint(0, input_shape[dim], input_shape, dtype=torch.long)
    return tensor, indices
