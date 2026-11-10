from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from operators.normalization.rmsnorm_gated.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Applies residual RMSNorm and a Swish gate to hidden states."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.eps = float(shape["eps"])
        self.prenorm = bool(shape["prenorm"])
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
        gate_input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden = x.to(self.compute_dtype) + residual.to(self.compute_dtype)
        weight_compute = weight.to(self.compute_dtype)
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden * torch.rsqrt(variance + self.eps)
        normed = normed * weight_compute
        gate = gate_input.to(self.compute_dtype)
        output = normed * (gate * torch.sigmoid(gate))

        output = output.to(self.output_dtype)
        if not self.prenorm:
            return output
        return output, hidden.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    hidden_dim = int(shape["hidden_dim"])
    tensor_shape = (batch_size, hidden_dim)
    x = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    gate_input = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    weight = torch.randn((hidden_dim,), dtype=torch.float32).to(dtype)
    residual = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return x, gate_input, weight, residual
