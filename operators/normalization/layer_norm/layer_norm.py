from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.normalization.layer_norm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Layer normalization using torch.nn.functional.layer_norm."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.normalized_shape = tuple(int(dim) for dim in shape["normalized_shape"])
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
        tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        output = F.layer_norm(
            tensor.to(self.compute_dtype),
            self.normalized_shape,
            weight.to(self.compute_dtype),
            bias.to(self.compute_dtype),
            self.eps,
        )
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    normalized_shape = tuple(int(dim) for dim in shape["normalized_shape"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    weight = torch.randn(normalized_shape, dtype=torch.float32).to(dtype)
    bias = torch.randn(normalized_shape, dtype=torch.float32).to(dtype)
    return tensor, weight, bias
