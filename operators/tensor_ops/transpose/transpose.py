from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.transpose.helpers import (
    _normalize_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Swap two tensor dimensions using torch.transpose."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        tensor_shape = tuple(int(dim) for dim in shape["shape"])
        self.dim0 = _normalize_dim(shape["dim0"], tensor_shape, "dim0")
        self.dim1 = _normalize_dim(shape["dim1"], tensor_shape, "dim1")
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.transpose(tensor, self.dim0, self.dim1).contiguous().to(
            self.output_dtype
        )


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])

    if dtype in (torch.int32, torch.int64):
        tensor = torch.randint(0, 1024, tensor_shape, dtype=dtype)
    else:
        tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (tensor,)
