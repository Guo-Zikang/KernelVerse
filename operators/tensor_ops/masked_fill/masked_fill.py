from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.masked_fill.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Fill masked tensor positions using torch.Tensor.masked_fill."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.fill_value = float(shape.get("fill_value", 0.0))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        output = tensor_compute.masked_fill(mask.bool(), self.fill_value)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    mask_probability = float(shape.get("mask_probability", 0.5))

    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    mask = torch.rand(tensor_shape) < mask_probability
    return tensor, mask
