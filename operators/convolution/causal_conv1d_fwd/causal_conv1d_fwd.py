from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.convolution.causal_conv1d_fwd.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Applies a causal depthwise 1D convolution over sequence positions."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.silu_activation = bool(shape["silu_activation"])
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
        weight_compute = weight.to(self.compute_dtype).unsqueeze(1)
        bias_compute = bias.to(self.compute_dtype)
        width = int(weight.shape[1])
        x_padded = F.pad(x_compute, pad=(width - 1, 0))
        output = F.conv1d(
            x_padded,
            weight_compute,
            bias_compute,
            groups=int(x.shape[1]),
        )
        if self.silu_activation:
            output = F.silu(output)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    dim = int(shape["dim"])
    x = torch.randn(
        (int(shape["batch"]), dim, int(shape["seqlen"])),
        dtype=torch.float32,
    ).to(dtype)
    weight = torch.randn((dim, int(shape["width"])), dtype=torch.float32).to(dtype)
    if shape["use_bias"]:
        bias = torch.randn((dim,), dtype=torch.float32).to(dtype)
    else:
        bias = torch.zeros((dim,), dtype=dtype)
    return x, weight, bias
