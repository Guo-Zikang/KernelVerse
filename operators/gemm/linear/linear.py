from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.gemm.linear.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Linear projection using torch.nn.functional.linear."""

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
        tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        weight_compute = weight.to(self.compute_dtype)
        bias_compute = bias.to(self.compute_dtype)
        output = F.linear(tensor_compute, weight_compute, bias_compute)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    input_shape = tuple(int(dim) for dim in shape["input_shape"])
    in_features = input_shape[-1]
    out_features = int(shape["out_features"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    weight = torch.randn((out_features, in_features), dtype=torch.float32).to(dtype)
    bias = torch.randn((out_features,), dtype=torch.float32).to(dtype)
    return tensor, weight, bias
