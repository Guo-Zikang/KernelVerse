from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.paged_attention_v1.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference paged attention v1 decode kernel."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.scale = float(shape.get("softmax_scale", int(shape["head_dim"]) ** -0.5))
        self.output_dtype = _resolve_dtype(str(shape["output_dtype"]))
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else _resolve_dtype(str(shape["input_dtype"]))
        )

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        query_compute = query.to(self.compute_dtype)
        key_cache_compute = key_cache.to(self.compute_dtype)
        value_cache_compute = value_cache.to(self.compute_dtype)
        batch_size = query_compute.shape[0]
        output = torch.zeros_like(query_compute, dtype=torch.float32)

        for batch_idx in range(batch_size):
            seq_len = int(seq_lens[batch_idx].item())
            keys, values = self._gather_paged_kv(
                key_cache_compute,
                value_cache_compute,
                block_table[batch_idx],
                seq_len,
            )
            output[batch_idx] = self._compute_decode_attention(
                query_compute[batch_idx],
                keys,
                values,
            )

        return output.to(self.output_dtype)

    def _gather_paged_kv(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_k = key_cache[block_table_row.long()]
        selected_v = value_cache[block_table_row.long()]
        flat_k = selected_k.reshape(-1, key_cache.shape[1], key_cache.shape[-1])[:seq_len]
        flat_v = selected_v.reshape(-1, value_cache.shape[1], value_cache.shape[-1])[
            :seq_len
        ]
        return flat_k, flat_v

    def _compute_decode_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        num_q_heads = query.shape[0]
        num_kv_heads = keys.shape[1]
        group_size = num_q_heads // num_kv_heads
        output = torch.empty_like(query, dtype=torch.float32)

        for q_head in range(num_q_heads):
            kv_head = q_head // group_size
            q = query[q_head].float()
            k = keys[:, kv_head, :].float()
            v = values[:, kv_head, :].float()
            logits = torch.matmul(k, q) * self.scale
            probs = torch.softmax(logits, dim=-1)
            output[q_head] = torch.matmul(probs.unsqueeze(0), v).squeeze(0)

        return output


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    block_size = int(shape["block_size"])
    input_dtype = _resolve_dtype(str(shape["input_dtype"]))
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * blocks_per_seq

    query = torch.randn((batch_size, num_q_heads, head_dim), dtype=torch.float32).to(
        input_dtype
    )
    key_cache = torch.randn(
        (total_blocks, num_kv_heads, block_size, head_dim), dtype=torch.float32
    ).to(input_dtype)
    value_cache = torch.randn(
        (total_blocks, num_kv_heads, block_size, head_dim), dtype=torch.float32
    ).to(input_dtype)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
    block_table = torch.arange(total_blocks, dtype=torch.int32).view(
        batch_size, blocks_per_seq
    )
    return query, key_cache, value_cache, seq_lens, block_table
