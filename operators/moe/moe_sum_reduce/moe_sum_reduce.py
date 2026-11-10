from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_sum_reduce.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference MoE routed expert output sum with scaling."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.routed_scaling_factor = float(shape["routed_scaling_factor"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(self.compute_dtype).sum(dim=1) * self.routed_scaling_factor
        return y.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    x = torch.randn(
        (
            int(shape["num_tokens"]),
            int(shape["topk"]),
            int(shape["hidden_size"]),
        ),
        dtype=torch.float32,
    ).to(dtype)
    return (x,)
