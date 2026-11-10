from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.activation.mul_and_silu.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference value * silu(gate) activation fusion."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_compute = x.to(self.compute_dtype)
        half_dim = x_compute.shape[-1] // 2
        value = x_compute[..., :half_dim]
        gate = x_compute[..., half_dim:]
        return (value * F.silu(gate)).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    x = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (x,)
