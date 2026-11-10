from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.nonzero.helpers import (
    _resolve_index_dtype,
    _resolve_input_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference nonzero coordinate extraction."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.index_dtype = _resolve_index_dtype(shape["index_dtype"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x, as_tuple=False).to(self.index_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    input_dims = tuple(int(dim) for dim in shape["input_dims"])
    dtype = _resolve_input_dtype(shape["dtype"])
    target_nonzero_count = int(shape["target_nonzero_count"])
    numel = 1
    for dim in input_dims:
        numel *= dim

    flat = torch.zeros(numel, dtype=dtype)
    if target_nonzero_count > 0:
        positions = torch.randperm(numel, dtype=torch.int64)[:target_nonzero_count]
        if dtype == torch.bool:
            flat[positions] = True
        elif dtype in (torch.int32, torch.int64):
            flat[positions] = torch.randint(
                low=1,
                high=17,
                size=(target_nonzero_count,),
                dtype=dtype,
            )
        else:
            values = torch.randn(target_nonzero_count, dtype=torch.float32)
            values = torch.where(values == 0, torch.ones_like(values), values)
            flat[positions] = values.to(dtype)
    return (flat.reshape(input_dims),)
