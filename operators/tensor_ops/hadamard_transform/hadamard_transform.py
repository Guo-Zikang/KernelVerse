from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.hadamard_transform.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Fast Hadamard transform over the last tensor dimension."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.normalize = bool(shape["normalize"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.to(self.compute_dtype).clone()
        last_dim = out.shape[-1]
        width = 1
        while width < last_dim:
            out = out.reshape(*out.shape[:-1], -1, 2 * width)
            left = out[..., :, :width].clone()
            right = out[..., :, width : 2 * width].clone()
            out[..., :, :width] = left + right
            out[..., :, width : 2 * width] = left - right
            out = out.reshape(*tensor.shape[:-1], last_dim)
            width *= 2

        if self.normalize:
            out = out / (last_dim ** 0.5)
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (tensor,)
