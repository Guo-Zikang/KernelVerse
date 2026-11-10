from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.quantization.quantize_per_tensor.helpers import (
    _resolve_quantized_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Per-tensor affine quantization using torch.quantize_per_tensor."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.scale = float(shape["scale"])
        self.zero_point = int(shape["zero_point"])
        self.quantized_dtype = _resolve_quantized_dtype(shape["quantized_dtype"])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.quantize_per_tensor(
            tensor.to(torch.float32),
            self.scale,
            self.zero_point,
            self.quantized_dtype,
        )


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32)
    return (tensor,)
