from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.topk_weight_and_reduce_batched.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reduce batched local expert outputs selected by top-k routing."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.rank = int(shape["rank"])
        self.apply_router_weight_on_input = bool(shape["apply_router_weight_on_input"])
        self.output_dtype = _resolve_float_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = topk_ids.size(0)
        num_local_experts = fused_expert_output.size(0)
        hidden_dim = fused_expert_output.size(-1)
        output = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=self.compute_dtype,
            device=fused_expert_output.device,
        )

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts
        for expert_id in range(first_expert, last_expert):
            matching_tokens = topk_ids == expert_id
            token_mask = torch.any(matching_tokens, dim=1).flatten()
            rows = int(torch.count_nonzero(token_mask).item())
            if rows == 0:
                continue
            rhs = fused_expert_output[expert_id - first_expert, :rows, :].to(self.compute_dtype)
            if not self.apply_router_weight_on_input:
                rhs = rhs * topk_weights[matching_tokens].to(self.compute_dtype).view(rows, 1)
            output[token_mask] = output[token_mask] + rhs
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    topk = int(shape["topk"])
    num_local_experts = int(shape["num_local_experts"])
    hidden_dim = int(shape["hidden_dim"])
    rank = int(shape["rank"])
    dtype = _resolve_float_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    fused_expert_output = torch.randn(
        (num_local_experts, num_tokens, hidden_dim),
        dtype=torch.float32,
    ).to(dtype)
    topk_weights = _normalize_prob_rows(
        torch.rand((num_tokens, topk), dtype=torch.float32)
    ).to(dtype)
    total_experts = num_local_experts * (rank + 1)
    topk_ids = torch.topk(
        torch.rand((num_tokens, total_experts), dtype=torch.float32),
        k=topk,
        dim=-1,
        sorted=False,
    ).indices.to(index_dtype)
    return fused_expert_output, topk_weights, topk_ids


def _normalize_prob_rows(weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return weights / denom
