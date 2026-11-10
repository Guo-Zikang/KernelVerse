from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.mm_encoder_attention_sdpa.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Compute grouped SDPA for multimodal encoder attention."""

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

    def _stable_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        exp_sum = exp_scores.sum(dim=-1, keepdim=True)
        return exp_scores / exp_sum

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        value_compute = value.to(self.compute_dtype)
        batch_size, num_q_heads, _, _ = query_compute.shape
        _, num_kv_heads, seq_k, _ = key_compute.shape
        group_size = num_q_heads // num_kv_heads
        outputs = []
        scale = self.head_dim ** -0.5

        for kv_head in range(num_kv_heads):
            q_start = kv_head * group_size
            q_end = q_start + group_size
            q_group = query_compute[:, q_start:q_end]
            k_head = key_compute[:, kv_head : kv_head + 1]
            v_head = value_compute[:, kv_head : kv_head + 1]
            expanded_key = k_head.expand(batch_size, group_size, seq_k, k_head.shape[-1])
            expanded_value = v_head.expand(batch_size, group_size, seq_k, v_head.shape[-1])

            scores = torch.einsum("bhqd,bhkd->bhqk", q_group, expanded_key) * scale
            probs = self._stable_softmax(scores)
            outputs.append(torch.einsum("bhqk,bhkd->bhqd", probs, expanded_value))

        return torch.cat(outputs, dim=1).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    q = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    k = torch.randn(
        (batch_size, num_kv_heads, seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    v = torch.randn(
        (batch_size, num_kv_heads, seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    return q, k, v
