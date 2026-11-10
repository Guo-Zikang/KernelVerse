from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.disentangled_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """DeBERTa-style disentangled attention with content and relative position interactions."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.max_relative_position = int(shape["max_relative_position"])
        self.num_relative_positions = 2 * self.max_relative_position + 1
        self.d_model = self.num_heads * self.head_dim
        self.use_bias = bool(shape.get("bias", True))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        self.w_qc = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.w_kc = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.w_vc = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.w_qp = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_kp = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=self.use_bias)
        self.relative_position_embedding = nn.Embedding(
            self.num_relative_positions,
            self.d_model,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        q_c = self._linear(tensor, self.w_qc)
        k_c = self._linear(tensor, self.w_kc)
        v_c = self._linear(tensor, self.w_vc)

        rel_indices = self._make_relative_position_indices(seq_len, tensor.device)
        rel_weight = self.relative_position_embedding.weight.to(self.compute_dtype)
        rel_pos = F.embedding(rel_indices, rel_weight)
        q_p = self._linear(rel_pos, self.w_qp)
        k_p = self._linear(rel_pos, self.w_kp)

        q_c = q_c.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_c = k_c.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = v_c.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_p = q_p.view(seq_len, seq_len, self.num_heads, self.head_dim).permute(
            2,
            0,
            1,
            3,
        )
        k_p = k_p.view(seq_len, seq_len, self.num_heads, self.head_dim).permute(
            2,
            0,
            1,
            3,
        )

        score_c2c = torch.matmul(q_c, k_c.transpose(-2, -1))
        score_c2p = torch.einsum("bhid,hijd->bhij", q_c, k_p)
        score_p2c = torch.einsum("hijd,bhjd->bhij", q_p, k_c)
        scores = (score_c2c + score_c2p + score_p2c) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_c)

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

    def _make_relative_position_indices(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(seq_len, device=device).view(seq_len, 1)
        key_positions = torch.arange(seq_len, device=device).view(1, seq_len)
        relative_position = query_positions - key_positions
        relative_position = relative_position.clamp(
            min=-self.max_relative_position,
            max=self.max_relative_position,
        )
        return relative_position + self.max_relative_position


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
