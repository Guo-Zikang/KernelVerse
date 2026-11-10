from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.batched_moe_align_block_size.helpers import (
    _resolve_index_dtype,
    round_up,
    validate_shape,
)


def _align_moe_tokens(
    expert_num_tokens: torch.Tensor,
    max_tokens_per_batch: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = int(expert_num_tokens.numel())
    padded_tokens_per_expert = round_up(max_tokens_per_batch, block_size)
    max_num_tokens_padded = num_experts * padded_tokens_per_expert
    sentinel = num_experts * max_tokens_per_batch
    device = expert_num_tokens.device

    sorted_ids = torch.full((max_num_tokens_padded,), sentinel, dtype=torch.int32, device=device)
    expert_ids = torch.full((max_num_tokens_padded // block_size,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.zeros((1,), dtype=torch.int32, device=device)

    cursor = 0
    block_cursor = 0
    for expert_id, expert_nt in enumerate(expert_num_tokens.to(torch.int64)):
        count = int(expert_nt.item())
        token_offset = expert_id * max_tokens_per_batch

        if count > 0:
            sorted_ids[cursor: cursor + count] = torch.arange(count, dtype=torch.int32, device=device) + token_offset

        padded = round_up(count, block_size)
        if padded > 0:
            num_blocks = padded // block_size
            expert_ids[block_cursor: block_cursor + num_blocks] = expert_id
            block_cursor += num_blocks
        cursor += padded

    num_tokens_post_pad[0] = cursor
    return sorted_ids, expert_ids, num_tokens_post_pad


class Model(nn.Module):
    """Aligns batched MoE expert token ids to block-size boundaries."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.max_tokens_per_batch = int(shape["max_tokens_per_batch"])
        self.block_size = int(shape["block_size"])
        self.output_index_dtype = _resolve_index_dtype(shape["output_index_dtype"])

    def forward(
        self,
        expert_num_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_ids, expert_ids, num_tokens_post_pad = _align_moe_tokens(
            expert_num_tokens,
            self.max_tokens_per_batch,
            self.block_size,
        )
        return (
            sorted_ids.to(self.output_index_dtype),
            expert_ids.to(self.output_index_dtype),
            num_tokens_post_pad.to(self.output_index_dtype),
        )


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    max_tokens_per_batch = int(shape["max_tokens_per_batch"])
    num_experts = int(shape["num_experts"])
    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch + 1,
        size=(num_experts,),
        dtype=torch.int32,
    )
    if shape["simulate_empty_batches"] and num_experts > 0:
        zero_count = max(1, num_experts // 3)
        zero_experts = torch.randperm(num_experts)[:zero_count]
        expert_num_tokens[zero_experts] = 0
    return (expert_num_tokens,)
