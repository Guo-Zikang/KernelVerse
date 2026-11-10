from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.normalization.add_rmsnorm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Fuses residual addition and RMSNorm over the last dimension."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.eps = float(shape["eps"])
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
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        summed = x.to(self.compute_dtype) + residual.to(self.compute_dtype)
        weight_compute = weight.to(self.compute_dtype)
        variance = summed.pow(2).mean(dim=-1, keepdim=True)
        normalized = summed * torch.rsqrt(variance + self.eps)
        output = normalized * weight_compute
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    hidden_size = tensor_shape[-1]
    dtype = _resolve_dtype(shape["dtype"])
    x = (torch.rand(tensor_shape, dtype=torch.float32) * 2 - 1).to(dtype)
    residual = (torch.rand(tensor_shape, dtype=torch.float32) * 2 - 1).to(dtype)
    weight = torch.ones((hidden_size,), dtype=torch.float32).to(dtype)
    return x, residual, weight
