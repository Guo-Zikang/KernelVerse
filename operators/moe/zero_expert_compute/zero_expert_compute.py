from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.zero_expert_compute.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


def _normalize_prob_rows(weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return weights / denom


class Model(nn.Module):
    """Reference identity zero-expert contribution for routed tokens."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_experts = int(shape["num_experts"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        expert_indices: torch.Tensor,
        expert_scales: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        zero_mask = expert_indices >= self.num_experts
        scales = expert_scales.to(self.compute_dtype)
        zero_expert_scales = torch.where(zero_mask, scales, torch.zeros_like(scales))
        out = hidden_states.to(self.compute_dtype) * zero_expert_scales.sum(dim=-1, keepdim=True)
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    hidden_dim = int(shape["hidden_dim"])
    topk = int(shape["topk"])
    num_experts = int(shape["num_experts"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    expert_indices = torch.randint(
        0,
        num_experts + 1,
        (num_tokens, topk),
        dtype=index_dtype,
    )
    expert_scales = _normalize_prob_rows(torch.rand((num_tokens, topk), dtype=torch.float32)).to(dtype)
    return (expert_indices, expert_scales, hidden_states)
