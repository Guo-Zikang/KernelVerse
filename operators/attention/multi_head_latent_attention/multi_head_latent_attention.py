from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.multi_head_latent_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """DeepSeek-style MLA decode reference over a paged compressed KV cache."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_q_heads = int(shape["num_q_heads"])
        self.qk_head_dim = int(shape["qk_head_dim"])
        self.v_head_dim = int(shape["v_head_dim"])
        self.block_size = int(shape["block_size"])
        self.causal = bool(shape.get("causal", True))
        self.scale = float(shape.get("softmax_scale", self.qk_head_dim ** -0.5))
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
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        query_compute = query.to(self.compute_dtype)
        kv_cache_compute = kv_cache.to(self.compute_dtype)
        batch_size = query_compute.shape[0]
        query_seq_len = query_compute.shape[1]
        outputs = []

        for batch_idx in range(batch_size):
            cache_seq_len = int(cache_seqlens[batch_idx].item())
            kv = self._reconstruct_from_cache(
                kv_cache_compute,
                cache_seq_len,
                block_table[batch_idx],
            )
            k = kv[:, :, : self.qk_head_dim].unsqueeze(0).transpose(1, 2).float()
            v = kv[:, :, : self.v_head_dim].unsqueeze(0).transpose(1, 2).float()
            q = query_compute[batch_idx : batch_idx + 1].transpose(1, 2).float()

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if self.causal:
                row_idx = torch.arange(query_seq_len, device=query.device).unsqueeze(1)
                col_idx = torch.arange(cache_seq_len, device=query.device).unsqueeze(0)
                causal_mask = col_idx > (row_idx + cache_seq_len - query_seq_len)
                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0),
                    float("-inf"),
                )

            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            outputs.append(output.transpose(1, 2))

        return torch.cat(outputs, dim=0).to(self.output_dtype)

    def _reconstruct_from_cache(
        self,
        cache: torch.Tensor,
        cache_seq_len: int,
        block_table_row: torch.Tensor,
    ) -> torch.Tensor:
        num_blocks_needed = (cache_seq_len + self.block_size - 1) // self.block_size
        parts = []
        for block_idx in range(num_blocks_needed):
            physical_block = int(block_table_row[block_idx].item())
            if block_idx == num_blocks_needed - 1:
                remaining = cache_seq_len - block_idx * self.block_size
                parts.append(cache[physical_block, :remaining])
            else:
                parts.append(cache[physical_block])
        return torch.cat(parts, dim=0)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    query_seq_len = int(shape["query_seq_len"])
    num_q_heads = int(shape["num_q_heads"])
    qk_head_dim = int(shape["qk_head_dim"])
    num_blocks = int(shape["num_blocks"])
    block_size = int(shape["block_size"])
    max_blocks_per_seq = int(shape["max_blocks_per_seq"])
    cache_seq_len = int(shape["cache_seq_len"])
    dtype = _resolve_dtype(shape["dtype"])
    blocks_needed = (cache_seq_len + block_size - 1) // block_size

    query = torch.randn(
        (batch_size, query_seq_len, num_q_heads, qk_head_dim),
        dtype=torch.float32,
    ).to(dtype)
    kv_cache = torch.randn(
        (num_blocks, block_size, 1, qk_head_dim),
        dtype=torch.float32,
    ).to(dtype)
    block_table = torch.zeros(
        (batch_size, max_blocks_per_seq),
        dtype=torch.int32,
    )
    for batch_idx in range(batch_size):
        start = batch_idx * blocks_needed
        end = start + blocks_needed
        block_table[batch_idx, :blocks_needed] = torch.arange(
            start,
            end,
            dtype=torch.int32,
        )
    cache_seqlens = torch.full((batch_size,), cache_seq_len, dtype=torch.int32)
    return query, kv_cache, block_table, cache_seqlens
