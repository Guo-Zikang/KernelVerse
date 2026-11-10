from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.causal_multi_head_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Causal multi-head self-attention with Q/K/V/O projections."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.d_model = self.num_heads * self.head_dim
        self.use_bias = bool(shape.get("bias", True))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)

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

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device)
        )
        scores = scores.masked_fill(
            ~causal_mask.view(1, 1, seq_len, seq_len),
            torch.finfo(scores.dtype).min,
        )
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        bias = layer.bias.to(self.compute_dtype) if layer.bias is not None else None
        return F.linear(tensor, weight, bias)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    d_model = num_heads * head_dim
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32).to(dtype)
    return (tensor,)
