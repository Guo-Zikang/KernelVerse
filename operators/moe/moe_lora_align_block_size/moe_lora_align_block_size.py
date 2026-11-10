from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_lora_align_block_size.helpers import (
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference LoRA-aware MoE token alignment by expert blocks."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_experts = int(shape["num_experts"])
        self.block_size = int(shape["block_size"])
        self.max_loras = int(shape["max_loras"])
        self.max_num_tokens_padded = int(shape["max_num_tokens_padded"])
        self.max_num_m_blocks = int(shape["max_num_m_blocks"])
        self.output_index_dtype = _resolve_index_dtype(shape["output_index_dtype"])

    def forward(
        self,
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens, topk = topk_ids.shape
        flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
        flat_lora = token_lora_mapping.to(torch.int64).repeat_interleave(topk)
        token_positions = torch.arange(
            num_tokens,
            device=topk_ids.device,
            dtype=torch.int64,
        ).repeat_interleave(topk)

        sorted_token_ids = torch.full(
            (self.max_loras, self.max_num_tokens_padded),
            fill_value=num_tokens,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        expert_ids = torch.full(
            (self.max_loras, self.max_num_m_blocks),
            fill_value=-1,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        num_tokens_post_pad = torch.zeros(
            (self.max_loras,),
            dtype=torch.int64,
            device=topk_ids.device,
        )
        adapter_enabled = torch.zeros(
            (self.max_loras,),
            dtype=torch.int64,
            device=topk_ids.device,
        )
        lora_ids = torch.full(
            (self.max_loras,),
            fill_value=-1,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        valid_loras = torch.unique(token_lora_mapping.to(torch.int64), sorted=True)
        valid_loras = valid_loras[(valid_loras >= 0) & (valid_loras < self.max_loras)]

        for lora_slot, lora_id_tensor in enumerate(valid_loras.tolist()):
            lora_id = int(lora_id_tensor)
            adapter_enabled[lora_slot] = 1
            lora_ids[lora_slot] = lora_id
            mask = flat_lora == lora_id
            lora_experts = flat_topk_ids[mask]
            lora_tokens = token_positions[mask]
            cursor = 0
            block_cursor = 0

            for expert_idx in range(self.num_experts):
                matched = lora_tokens[lora_experts == expert_idx]
                real_count = int(matched.numel())
                padded_count = (
                    (real_count + self.block_size - 1) // self.block_size * self.block_size
                    if real_count > 0
                    else 0
                )
                if padded_count == 0:
                    continue
                end = min(cursor + padded_count, self.max_num_tokens_padded)
                copy_count = min(real_count, end - cursor)
                if copy_count > 0:
                    sorted_token_ids[lora_slot, cursor:cursor + copy_count] = matched[:copy_count]
                num_blocks = min(
                    padded_count // self.block_size,
                    self.max_num_m_blocks - block_cursor,
                )
                if num_blocks > 0:
                    expert_ids[lora_slot, block_cursor:block_cursor + num_blocks] = expert_idx
                    block_cursor += num_blocks
                cursor = min(cursor + padded_count, self.max_num_tokens_padded)
            num_tokens_post_pad[lora_slot] = cursor

        outputs = (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            adapter_enabled,
            lora_ids,
        )
        return tuple(t.to(self.output_index_dtype) for t in outputs)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    topk = int(shape["topk"])
    num_experts = int(shape["num_experts"])
    max_loras = int(shape["max_loras"])
    input_index_dtype = _resolve_index_dtype(shape["input_index_dtype"])
    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, topk),
        dtype=input_index_dtype,
    )
    token_lora_mapping = torch.randint(
        low=0,
        high=max_loras,
        size=(num_tokens,),
        dtype=torch.int32,
    )
    return topk_ids, token_lora_mapping
