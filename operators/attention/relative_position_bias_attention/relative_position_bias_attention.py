from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.relative_position_bias_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Multi-head attention with T5-style relative position bucket bias."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.num_buckets = int(shape.get("num_buckets", 32))
        self.max_distance = int(shape.get("max_distance", 128))
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
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

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
        scores = scores + self._make_relative_position_bias(seq_len, scores.device)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _make_relative_position_bias(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = context_position - memory_position
        bucket = self._relative_position_bucket(relative_position)
        bias_weight = self.relative_attention_bias.weight.to(self.compute_dtype)
        values = F.embedding(bucket, bias_weight)
        return values.permute(2, 0, 1).unsqueeze(0)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        half_buckets = self.num_buckets // 2
        sign_bucket = (relative_position < 0).long() * half_buckets
        distance = torch.abs(relative_position)

        max_exact = max(1, half_buckets // 2)
        is_small = distance < max_exact
        safe_distance = torch.clamp(distance, min=max_exact)
        log_ratio = torch.log(safe_distance.float() / max_exact)
        log_denom = torch.log(
            torch.tensor(self.max_distance / max_exact, device=distance.device)
        )
        large_bucket = max_exact + (
            log_ratio / log_denom * (half_buckets - max_exact)
        ).long()
        large_bucket = torch.minimum(
            large_bucket,
            torch.full_like(large_bucket, half_buckets - 1),
        )
        bucket = torch.where(is_small, distance.long(), large_bucket)
        return bucket + sign_bucket

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
