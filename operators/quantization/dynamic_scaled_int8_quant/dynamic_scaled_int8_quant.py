from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.quantization.dynamic_scaled_int8_quant.helpers import (
    INT8_QMAX,
    MIN_SCALING_FACTOR,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Dynamically quantizes a tensor to INT8 values with one global scale."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = (x.abs().amax().reshape(1, 1) / INT8_QMAX).to(torch.float32)
        scale = scale.clamp(min=MIN_SCALING_FACTOR)
        q = torch.round((x.float() / scale).clamp(-INT8_QMAX, INT8_QMAX)).to(torch.int8)
        azp = torch.zeros((1,), dtype=torch.int32, device=x.device)
        return q, scale.reshape(1), azp


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    x = torch.randn(tuple(shape["dims"]), dtype=torch.float32).to(dtype)
    return (x,)
