from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.mul.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference elementwise multiply with broadcasting."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            and self.output_dtype in (torch.float16, torch.bfloat16)
            else self.output_dtype
        )

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        output = lhs.to(self.compute_dtype) * rhs.to(self.compute_dtype)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    lhs_dims = tuple(int(dim) for dim in shape["lhs_dims"])
    rhs_dims = tuple(int(dim) for dim in shape["rhs_dims"])

    if dtype in (torch.int32, torch.int64):
        lhs = torch.randint(low=-128, high=128, size=lhs_dims, dtype=dtype)
        rhs = torch.randint(low=-8, high=8, size=rhs_dims, dtype=dtype)
    else:
        lhs = torch.randn(lhs_dims, dtype=torch.float32).to(dtype)
        rhs = torch.randn(rhs_dims, dtype=torch.float32).to(dtype)
    return lhs, rhs
