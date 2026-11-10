from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_align_block_size.helpers import (
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference MoE token alignment by expert blocks."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_experts = int(shape["num_experts"])
        self.block_size = int(shape["block_size"])
        self.pad_sorted_token_ids = bool(shape["pad_sorted_token_ids"])
        self.output_index_dtype = _resolve_index_dtype(shape["output_index_dtype"])

    def forward(
        self,
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
        numel = int(flat_topk_ids.numel())
        sentinel = numel
        sorted_token_segments = []
        expert_ids = []
        cumsum_values = [0]

        for expert_idx in range(self.num_experts):
            matched_positions = torch.nonzero(
                flat_topk_ids == expert_idx,
                as_tuple=False,
            ).flatten()
            real_count = int(matched_positions.numel())
            padded_count = (
                (real_count + self.block_size - 1) // self.block_size * self.block_size
            )
            if padded_count > 0:
                segment = torch.full(
                    (padded_count,),
                    fill_value=sentinel,
                    dtype=torch.int64,
                    device=topk_ids.device,
                )
                if real_count > 0:
                    segment[:real_count] = matched_positions
                    if self.pad_sorted_token_ids and real_count < padded_count:
                        segment[real_count:] = matched_positions[-1]
                sorted_token_segments.append(segment)
                expert_ids.extend([expert_idx] * (padded_count // self.block_size))
            cumsum_values.append(cumsum_values[-1] + padded_count)

        if sorted_token_segments:
            sorted_token_ids = torch.cat(sorted_token_segments, dim=0)
        else:
            sorted_token_ids = torch.empty((0,), dtype=torch.int64, device=topk_ids.device)
        expert_ids_tensor = torch.tensor(
            expert_ids,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        num_tokens_post_pad = torch.tensor(
            [cumsum_values[-1]],
            dtype=torch.int64,
            device=topk_ids.device,
        )
        cumsum_buffer = torch.tensor(
            cumsum_values,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        return (
            sorted_token_ids.to(self.output_index_dtype),
            expert_ids_tensor.to(self.output_index_dtype),
            num_tokens_post_pad.to(self.output_index_dtype),
            cumsum_buffer.to(self.output_index_dtype),
        )


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    topk = int(shape["topk"])
    num_experts = int(shape["num_experts"])
    input_index_dtype = _resolve_index_dtype(shape["input_index_dtype"])
    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, topk),
        dtype=input_index_dtype,
    )
    return (topk_ids,)
