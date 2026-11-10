from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.repeat_interleave.helpers import (
    _normalize_dim,
    _resolve_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Repeat tensor elements using torch.repeat_interleave."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        tensor_shape = tuple(int(dim) for dim in shape["shape"])
        dim = _resolve_dim(shape.get("dim", None))
        self.dim = None if dim is None else _normalize_dim(dim, tensor_shape)
        self.repeats = int(shape["repeats"])
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = torch.repeat_interleave(tensor, self.repeats, dim=self.dim)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])

    if dtype in (torch.int32, torch.int64):
        tensor = torch.randint(0, 1024, tensor_shape, dtype=dtype)
    else:
        tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (tensor,)
