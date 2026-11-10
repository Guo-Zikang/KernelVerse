from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.quantization.per_token_group_quant_int8.helpers import (
    _resolve_dtype,
    validate_shape,
)

INT8_QMAX = 127.0


class Model(nn.Module):
    """Reference per-token per-group INT8 quantization."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.group_size = int(shape["group_size"])
        self.eps = float(shape["eps"])

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens, hidden_dim = input_tensor.shape
        num_groups = hidden_dim // self.group_size
        grouped = input_tensor.float().view(num_tokens, num_groups, self.group_size)
        scales = (grouped.abs().amax(dim=-1) / INT8_QMAX).clamp_min(self.eps)
        q = torch.round(grouped / scales.unsqueeze(-1)).clamp(
            -INT8_QMAX,
            INT8_QMAX,
        ).to(torch.int8)
        return q.view(num_tokens, hidden_dim), scales.float()


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    input_tensor = torch.randn(
        (int(shape["num_tokens"]), int(shape["hidden_dim"])),
        dtype=torch.float32,
    ).to(dtype)
    return (input_tensor,)
