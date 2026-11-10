from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.quantization.static_scaled_int8_quant.helpers import (
    INT8_QMAX,
    MIN_SCALE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Quantize a tensor to INT8 using a static scale and zero point."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.input_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.input_dtype
        )

    def forward(self, x: torch.Tensor, scale: torch.Tensor, azp: torch.Tensor) -> torch.Tensor:
        scale_value = scale.reshape(1).to(self.compute_dtype).clamp(min=MIN_SCALE)
        centered = x.to(self.compute_dtype) / scale_value
        centered = centered + azp.to(self.compute_dtype).reshape(1)
        return torch.round(centered).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    x = torch.randn(tuple(shape["dims"]), dtype=torch.float32).to(dtype)
    scale = torch.tensor([0.125], dtype=torch.float32)
    azp = torch.tensor([0], dtype=torch.int32)
    return x, scale, azp
