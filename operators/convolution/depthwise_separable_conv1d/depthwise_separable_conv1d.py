from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.convolution.depthwise_separable_conv1d.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Applies depthwise and pointwise 1D convolutions to sequence features."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.padding = int(shape["padding"])
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
        dw_weight: torch.Tensor,
        dw_bias: torch.Tensor,
        pw_weight: torch.Tensor,
        pw_bias: torch.Tensor,
    ) -> torch.Tensor:
        x_compute = x.to(self.compute_dtype).permute(0, 2, 1)
        dw_output = F.conv1d(
            x_compute,
            dw_weight.to(self.compute_dtype),
            dw_bias.to(self.compute_dtype),
            padding=self.padding,
            groups=int(x.shape[2]),
        )
        output = F.conv1d(
            dw_output,
            pw_weight.to(self.compute_dtype),
            pw_bias.to(self.compute_dtype),
        )
        return output.permute(0, 2, 1).to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    input_dim = int(shape["input_dim"])
    hidden_channels = input_dim * int(shape["depthwise_multiplier"])
    kernel_size = int(shape["kernel_size"])
    out_channels = int(shape["out_channels"])

    x = torch.randn(
        (int(shape["batch_size"]), int(shape["seq_len"]), input_dim),
        dtype=torch.float32,
    ).to(dtype)
    dw_weight = torch.randn(
        (hidden_channels, 1, kernel_size),
        dtype=torch.float32,
    ).to(dtype)
    dw_bias = torch.randn((hidden_channels,), dtype=torch.float32).to(dtype)
    pw_weight = torch.randn(
        (out_channels, hidden_channels, 1),
        dtype=torch.float32,
    ).to(dtype)
    pw_bias = torch.randn((out_channels,), dtype=torch.float32).to(dtype)
    return x, dw_weight, dw_bias, pw_weight, pw_bias
