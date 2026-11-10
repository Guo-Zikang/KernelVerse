from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.scatter_add.helpers import (
    _normalize_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Add source values into a tensor using torch.scatter_add."""

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

    def forward(
        self,
        tensor: torch.Tensor,
        index: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        src_compute = src.to(self.compute_dtype)
        output = torch.scatter_add(tensor_compute, self.dim, index.long(), src_compute)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    input_shape = tuple(int(dim) for dim in shape["input_shape"])
    dim = _normalize_dim(shape["dim"], input_shape)
    dtype = _resolve_dtype(shape["dtype"])

    tensor = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    index = torch.randint(0, input_shape[dim], input_shape, dtype=torch.long)
    src = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    return tensor, index, src
