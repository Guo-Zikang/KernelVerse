from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.chunk.helpers import (
    _normalize_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Split a tensor into chunks using torch.chunk."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        input_shape = tuple(int(dim) for dim in shape["input_shape"])
        self.dim = _normalize_dim(shape["dim"], input_shape)
        self.chunks = int(shape["chunks"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        tensor_compute = tensor.to(self.compute_dtype)
        outputs = torch.chunk(tensor_compute, self.chunks, dim=self.dim)
        return tuple(
            output.to(self.output_dtype).clone(memory_format=torch.contiguous_format)
            for output in outputs
        )


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    input_shape = tuple(int(dim) for dim in shape["input_shape"])
    dtype = _resolve_dtype(shape["dtype"])

    tensor = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    return (tensor,)
