from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.quantization.dequantize.helpers import (
    _resolve_quantized_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Dequantize a quantized tensor using torch.Tensor.dequantize."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.dequantize()


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32)
    scale = float(shape["scale"])
    zero_point = int(shape["zero_point"])
    quantized_dtype = _resolve_quantized_dtype(shape["quantized_dtype"])

    if shape["qscheme"] == "per_tensor":
        quantized = torch.quantize_per_tensor(
            tensor,
            scale,
            zero_point,
            quantized_dtype,
        )
    else:
        axis = int(shape["axis"])
        if axis < 0:
            axis += len(tensor_shape)
        channel_count = tensor_shape[axis]
        scales = torch.full((channel_count,), scale, dtype=torch.float64)
        zero_points = torch.full((channel_count,), zero_point, dtype=torch.int64)
        quantized = torch.quantize_per_channel(
            tensor,
            scales,
            zero_points,
            axis,
            quantized_dtype,
        )
    return (quantized,)
