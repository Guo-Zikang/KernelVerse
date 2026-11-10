from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_unpermute.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference MoE expert output unpermutation and weighted combine."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_float_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        permuted_hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, topk = topk_weights.shape
        hidden_dim = permuted_hidden_states.shape[-1]
        restored = permuted_hidden_states.index_select(
            0,
            inv_permuted_idx.to(torch.int64),
        )
        restored = restored.view(num_tokens, topk, hidden_dim)
        weighted = (
            restored.to(self.compute_dtype)
            * topk_weights.to(self.compute_dtype).unsqueeze(-1)
        )
        return weighted.sum(dim=1).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    hidden_dim = int(shape["hidden_dim"])
    topk = int(shape["topk"])
    num_experts = int(shape["num_experts"])
    dtype = _resolve_float_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, topk),
        dtype=torch.int32,
    )
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_topk_ids, stable=True)
    token_rows = torch.div(order, topk, rounding_mode="floor")
    permuted_hidden_states = hidden_states.index_select(
        0,
        token_rows.to(torch.int64),
    ).reshape(-1, hidden_dim)
    inv_permuted_idx = torch.empty(
        (flat_topk_ids.numel(),),
        dtype=torch.int64,
    )
    inv_permuted_idx[order] = torch.arange(order.numel(), dtype=torch.int64)

    topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32)
    denom = topk_weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    topk_weights = (topk_weights / denom).to(dtype)
    return permuted_hidden_states, topk_weights, inv_permuted_idx.to(index_dtype)
