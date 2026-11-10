from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.paged_attention_v2.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference paged attention v2 decode kernel with partitioned softmax merge."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.scale = float(shape.get("softmax_scale", int(shape["head_dim"]) ** -0.5))
        self.partition_size = int(shape["partition_size"])
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_compute = query.to(self.compute_dtype)
        key_cache_compute = key_cache.to(self.compute_dtype)
        value_cache_compute = value_cache.to(self.compute_dtype)
        batch_size = query_compute.shape[0]
        num_q_heads = query_compute.shape[1]
        head_dim = query_compute.shape[2]
        max_seq_len = int(seq_lens.max().item())
        max_num_partitions = (max_seq_len + self.partition_size - 1) // self.partition_size

        output = torch.zeros_like(query_compute, dtype=self.compute_dtype)
        exp_sums = torch.zeros(
            (batch_size, num_q_heads, max_num_partitions),
            dtype=self.compute_dtype,
            device=query_compute.device,
        )
        max_logits = torch.full(
            (batch_size, num_q_heads, max_num_partitions),
            fill_value=-float("inf"),
            dtype=self.compute_dtype,
            device=query_compute.device,
        )
        tmp_out = torch.zeros(
            (batch_size, num_q_heads, max_num_partitions, head_dim),
            dtype=self.compute_dtype,
            device=query_compute.device,
        )

        for batch_idx in range(batch_size):
            seq_len = int(seq_lens[batch_idx].item())
            keys, values = self._gather_paged_kv(
                key_cache_compute,
                value_cache_compute,
                block_table[batch_idx],
                seq_len,
            )
            batch_out, batch_exp_sums, batch_max_logits, batch_tmp_out = (
                self._compute_partitioned_decode_attention(
                    query_compute[batch_idx],
                    keys,
                    values,
                )
            )
            num_partitions = batch_exp_sums.shape[1]
            output[batch_idx] = batch_out
            exp_sums[batch_idx, :, :num_partitions] = batch_exp_sums
            max_logits[batch_idx, :, :num_partitions] = batch_max_logits
            tmp_out[batch_idx, :, :num_partitions, :] = batch_tmp_out

        return (
            output.to(self.output_dtype),
            exp_sums.to(self.output_dtype),
            max_logits.to(self.output_dtype),
            tmp_out.to(self.output_dtype),
        )

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

    def _compute_partitioned_decode_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_q_heads, head_dim = query.shape
        seq_len = keys.shape[0]
        num_partitions = (seq_len + self.partition_size - 1) // self.partition_size
        num_kv_heads = keys.shape[1]
        group_size = num_q_heads // num_kv_heads
        exp_sums = torch.zeros(
            (num_q_heads, num_partitions),
            dtype=self.compute_dtype,
            device=query.device,
        )
        max_logits = torch.full(
            (num_q_heads, num_partitions),
            fill_value=-float("inf"),
            dtype=self.compute_dtype,
            device=query.device,
        )
        tmp_out = torch.zeros(
            (num_q_heads, num_partitions, head_dim),
            dtype=self.compute_dtype,
            device=query.device,
        )
        output = torch.zeros(
            (num_q_heads, head_dim),
            dtype=self.compute_dtype,
            device=query.device,
        )

        for q_head in range(num_q_heads):
            kv_head = q_head // group_size
            q = query[q_head].to(self.compute_dtype)
            partition_outputs = []
            partition_sums = []
            partition_maxes = []

            for partition_idx in range(num_partitions):
                start = partition_idx * self.partition_size
                end = min(start + self.partition_size, seq_len)
                k = keys[start:end, kv_head, :].to(self.compute_dtype)
                v = values[start:end, kv_head, :].to(self.compute_dtype)
                logits = torch.matmul(k, q) * self.scale
                local_max = logits.max()
                exp_logits = torch.exp(logits - local_max)
                local_sum = exp_logits.sum()
                weighted = torch.matmul(exp_logits.unsqueeze(0), v).squeeze(0)
                max_logits[q_head, partition_idx] = local_max
                exp_sums[q_head, partition_idx] = local_sum
                tmp_out[q_head, partition_idx] = weighted / (local_sum + 1e-6)
                partition_outputs.append(weighted)
                partition_sums.append(local_sum)
                partition_maxes.append(local_max)

            global_max = torch.stack(partition_maxes).max()
            global_sum = torch.stack(
                [
                    part_sum * torch.exp(part_max - global_max)
                    for part_sum, part_max in zip(partition_sums, partition_maxes)
                ]
            ).sum()
            merged = torch.zeros((head_dim,), dtype=self.compute_dtype, device=query.device)
            for weighted, part_max in zip(partition_outputs, partition_maxes):
                merged += weighted * torch.exp(part_max - global_max)
            output[q_head] = merged / (global_sum + 1e-6)

        return output, exp_sums, max_logits, tmp_out


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
