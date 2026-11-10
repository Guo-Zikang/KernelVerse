from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.convolution.conv3d.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Applies a 3D convolution to spatiotemporal feature maps."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        groups = int(shape.get("groups", 1))
        self.stride = int(shape.get("stride", 1))
        self.padding = int(shape.get("padding", 0))
        self.dilation = int(shape.get("dilation", 1))
        self.groups = groups
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        x_compute = x.to(self.compute_dtype)
        weight_compute = weight.to(self.compute_dtype)
        bias_compute = bias.to(self.compute_dtype)
        output = F.conv3d(
            x_compute,
            weight_compute,
            bias_compute,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    groups = int(shape.get("groups", 1))
    kernel_size = int(shape["kernel_size"])
    input_shape = (
        int(shape["batch_size"]),
        int(shape["in_channels"]),
        int(shape["time"]),
        int(shape["height"]),
        int(shape["width"]),
    )
    weight_shape = (
        int(shape["out_channels"]),
        int(shape["in_channels"]) // groups,
        kernel_size,
        kernel_size,
        kernel_size,
    )
    x = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    weight = torch.randn(weight_shape, dtype=torch.float32).to(dtype)
    bias = torch.randn((int(shape["out_channels"]),), dtype=torch.float32).to(dtype)
    return x, weight, bias
