from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.gemm.bmm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Batched matrix multiplication using torch.bmm."""

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

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        tensor_a = tensor_a.to(self.compute_dtype)
        tensor_b = tensor_b.to(self.compute_dtype)
        output = torch.bmm(tensor_a, tensor_b)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    m_value = int(shape["M"])
    n_value = int(shape["N"])
    k_value = int(shape["K"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor_a = torch.randn((batch_size, m_value, k_value), dtype=torch.float32).to(dtype)
    tensor_b = torch.randn((batch_size, k_value, n_value), dtype=torch.float32).to(dtype)
    return tensor_a, tensor_b
