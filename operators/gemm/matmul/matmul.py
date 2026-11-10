from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.gemm.matmul.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Dense matrix multiplication using torch.matmul."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(tensor_a, tensor_b)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    m_value = int(shape["M"])
    n_value = int(shape["N"])
    k_value = int(shape["K"])
    dtype = _resolve_dtype(shape["dtype"])
    tensor_a = torch.randn((m_value, k_value), dtype=torch.float32).to(dtype)
    tensor_b = torch.randn((k_value, n_value), dtype=torch.float32).to(dtype)
    return tensor_a, tensor_b

