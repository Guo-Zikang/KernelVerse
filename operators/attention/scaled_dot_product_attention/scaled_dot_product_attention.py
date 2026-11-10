from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.scaled_dot_product_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k))V."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_dim = int(shape["head_dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query = query.to(self.compute_dtype)
        key = key.to(self.compute_dtype)
        value = value.to(self.compute_dtype)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    dims = (batch_size, num_heads, seq_len, head_dim)
    query = torch.randn(dims, dtype=torch.float32).to(dtype)
    key = torch.randn(dims, dtype=torch.float32).to(dtype)
    value = torch.randn(dims, dtype=torch.float32).to(dtype)
    return (query, key, value)
