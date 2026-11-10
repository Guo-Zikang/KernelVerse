from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.quantization.quantize_per_channel.helpers import (
    _resolve_quantized_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Per-channel affine quantization using torch.quantize_per_channel."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        tensor_shape = shape["shape"]
        axis = int(shape["axis"])
        if axis < 0:
            axis += len(tensor_shape)
        self.axis = axis
        self.scale = float(shape["scale"])
        self.zero_point = int(shape["zero_point"])
        self.quantized_dtype = _resolve_quantized_dtype(shape["quantized_dtype"])
        channel_count = int(tensor_shape[axis])
        channel_indices = torch.arange(channel_count, dtype=torch.float64)
        scale_multipliers = 1.0 + (channel_indices.remainder(8) / 8.0)
        self.register_buffer("scales", self.scale * scale_multipliers)
        self.register_buffer(
            "zero_points",
            torch.full((channel_count,), self.zero_point, dtype=torch.int64),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        quantized = torch.quantize_per_channel(
            tensor.to(torch.float32),
            self.scales,
            self.zero_points,
            self.axis,
            self.quantized_dtype,
        )
        return quantized.int_repr()


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32)
    return (tensor,)
