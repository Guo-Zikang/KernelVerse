from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.normalization.l2_norm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """L2 normalization over the last tensor dimension."""

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

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        norm = tensor_compute.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        output = tensor_compute / norm
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    return (tensor,)
