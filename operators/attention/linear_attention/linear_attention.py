from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.linear_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Linear attention using feature maps and K^T V reordering."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.d_model = self.num_heads * self.head_dim
        self.feature_map = shape.get("feature_map", "elu")
        self.eps = float(shape.get("eps", 1e-6))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        q = self._linear(tensor, self.w_q)
        k = self._linear(tensor, self.w_k)
        v = self._linear(tensor, self.w_v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(q)
        k = self._feature_map(k)

        kv = torch.matmul(k.transpose(-2, -1), v)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).unsqueeze(-1)
        normalizer = torch.reciprocal(normalizer + self.eps)
        output = torch.matmul(q, kv) * normalizer

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _feature_map(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return F.elu(tensor) + 1
        if self.feature_map == "relu":
            return F.relu(tensor)
        return tensor

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        bias = layer.bias.to(self.compute_dtype) if layer.bias is not None else None
        return F.linear(tensor, weight, bias)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    d_model = num_heads * head_dim
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32).to(dtype)
    return (tensor,)
