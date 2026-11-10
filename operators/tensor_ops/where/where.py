from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.where.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Select values using torch.where."""

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

    def forward(
        self,
        condition: torch.Tensor,
        tensor: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        other_compute = other.to(self.compute_dtype)
        output = torch.where(condition.bool(), tensor_compute, other_compute)
        return output.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    condition_probability = float(shape.get("condition_probability", 0.5))

    condition = torch.rand(tensor_shape) < condition_probability
    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    other = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return condition, tensor, other
