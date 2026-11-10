from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_permute.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference MoE token hidden-state permutation by expert id."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_experts = int(shape["num_experts"])
        self.index_dtype = _resolve_index_dtype(shape["index_dtype"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens, hidden_dim = hidden_states.shape
        topk = topk_ids.shape[1]
        flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)

        order = torch.argsort(flat_topk_ids, stable=True)
        token_rows = torch.div(order, topk, rounding_mode="floor")
        permuted_hidden_states = hidden_states.index_select(
            0,
            token_rows.to(torch.int64),
        ).reshape(-1, hidden_dim)

        counts = torch.bincount(flat_topk_ids, minlength=self.num_experts).to(torch.int64)
        expert_first_token_offset = torch.zeros(
            (self.num_experts + 1,),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        expert_first_token_offset[1:] = torch.cumsum(counts, dim=0)
        inv_permuted_idx = torch.empty(
            (flat_topk_ids.numel(),),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        inv_permuted_idx[order] = torch.arange(
            order.numel(),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        permuted_idx = order.to(torch.int32)
        return (
            permuted_hidden_states,
            expert_first_token_offset.to(self.index_dtype),
            inv_permuted_idx.to(self.index_dtype),
            permuted_idx.to(self.index_dtype),
        )


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    hidden_dim = int(shape["hidden_dim"])
    topk = int(shape["topk"])
    num_experts = int(shape["num_experts"])
    dtype = _resolve_float_dtype(shape["dtype"])
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, topk),
        dtype=torch.int32,
    )
    return hidden_states, topk_ids
