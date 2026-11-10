from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.flash_decoding_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Flash decoding attention reference with chunked KV softmax merge outputs."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_dim = int(shape["head_dim"])
        self.chunk_size = int(shape["chunk_size"])
        self.causal = bool(shape.get("causal", False))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        self.scale = float(shape.get("softmax_scale", self.head_dim ** -0.5))
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_compute = query.to(self.compute_dtype)
        key_compute = key_cache.to(self.compute_dtype)
        value_compute = value_cache.to(self.compute_dtype)
        kv_seq_len = key_compute.shape[2]

        partial_maxes = []
        partial_exp_sums = []
        partial_outputs = []

        for kv_start in range(0, kv_seq_len, self.chunk_size):
            kv_end = min(kv_start + self.chunk_size, kv_seq_len)
            key_chunk = key_compute[:, :, kv_start:kv_end, :]
            value_chunk = value_compute[:, :, kv_start:kv_end, :]

            scores = torch.matmul(query_compute, key_chunk.transpose(-2, -1))
            scores = scores * self.scale
            if self.causal:
                scores = self._apply_causal_mask(scores, kv_start, kv_end, kv_seq_len)

            partial_max = scores.max(dim=-1).values
            valid_chunk = torch.isfinite(partial_max)
            exp_scores = torch.exp(scores - partial_max.unsqueeze(-1))
            exp_scores = torch.where(
                valid_chunk.unsqueeze(-1),
                exp_scores,
                torch.zeros_like(exp_scores),
            )
            partial_exp_sum = exp_scores.sum(dim=-1)
            weighted_output = torch.matmul(exp_scores, value_chunk)
            partial_output = weighted_output / partial_exp_sum.clamp(min=1e-6).unsqueeze(
                -1
            )

            partial_maxes.append(partial_max)
            partial_exp_sums.append(partial_exp_sum)
            partial_outputs.append(partial_output)

        max_stack = torch.stack(partial_maxes, dim=-1)
        exp_sum_stack = torch.stack(partial_exp_sums, dim=-1)
        output_stack = torch.stack(partial_outputs, dim=-2)

        global_max = max_stack.max(dim=-1).values
        merge_weights = exp_sum_stack * torch.exp(max_stack - global_max.unsqueeze(-1))
        global_exp_sum = merge_weights.sum(dim=-1)
        output = (output_stack * merge_weights.unsqueeze(-1)).sum(dim=-2)
        output = output / global_exp_sum.clamp(min=1e-6).unsqueeze(-1)
        return (
            output.to(self.output_dtype),
            exp_sum_stack.to(self.output_dtype),
            max_stack.to(self.output_dtype),
            output_stack.to(self.output_dtype),
        )

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        kv_start: int,
        kv_end: int,
        kv_seq_len: int,
    ) -> torch.Tensor:
        query_seq_len = scores.shape[-2]
        query_positions = torch.arange(query_seq_len, device=scores.device).unsqueeze(1)
        key_positions = torch.arange(kv_start, kv_end, device=scores.device).unsqueeze(0)
        absolute_query_positions = query_positions + kv_seq_len - query_seq_len
        mask = key_positions > absolute_query_positions
        return scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    query_seq_len = int(shape["query_seq_len"])
    kv_seq_len = int(shape["kv_seq_len"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    query = torch.randn(
        (batch_size, num_heads, query_seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    key_cache = torch.randn(
        (batch_size, num_heads, kv_seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    value_cache = torch.randn(
        (batch_size, num_heads, kv_seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    return query, key_cache, value_cache
