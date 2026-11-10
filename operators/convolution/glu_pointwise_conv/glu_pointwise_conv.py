from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.convolution.glu_pointwise_conv.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Applies a convolution followed by a gated linear unit."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        output_dim = int(shape["output_dim"])
        kernel_size = int(shape["kernel_size"])
        self.output_dim = output_dim
        self.bias_in_glu = bool(shape["bias_in_glu"])
        self.causal = bool(shape["causal"])
        self.padding = kernel_size - 1 if self.causal else (kernel_size - 1) // 2
        self.causal_trim = self.padding if self.causal else 0
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
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        glu_bias_left: torch.Tensor,
        glu_bias_right: torch.Tensor,
    ) -> torch.Tensor:
        conv_input = x.to(self.compute_dtype).permute(0, 2, 1)
        conv_output = F.conv1d(
            conv_input,
            conv_weight.to(self.compute_dtype),
            conv_bias.to(self.compute_dtype),
            padding=self.padding,
        )
        if self.causal_trim > 0:
            conv_output = conv_output[..., :-self.causal_trim]

        left = conv_output[:, : self.output_dim, :]
        right = conv_output[:, self.output_dim :, :]
        if self.bias_in_glu:
            left = left + glu_bias_left.to(self.compute_dtype)
            right = right + glu_bias_right.to(self.compute_dtype)
        output = left * torch.sigmoid(right)
        return output.permute(0, 2, 1).to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    input_dim = int(shape["input_dim"])
    output_dim = int(shape["output_dim"])
    kernel_size = int(shape["kernel_size"])
    x = torch.randn(
        (int(shape["batch_size"]), int(shape["seq_len"]), input_dim),
        dtype=torch.float32,
    ).to(dtype)
    conv_weight = torch.randn(
        (output_dim * 2, input_dim, kernel_size),
        dtype=torch.float32,
    ).to(dtype)
    conv_bias = torch.randn((output_dim * 2,), dtype=torch.float32).to(dtype)
    glu_bias_left = torch.zeros((1, output_dim, 1), dtype=dtype)
    glu_bias_right = torch.zeros((1, output_dim, 1), dtype=dtype)
    return x, conv_weight, conv_bias, glu_bias_left, glu_bias_right
