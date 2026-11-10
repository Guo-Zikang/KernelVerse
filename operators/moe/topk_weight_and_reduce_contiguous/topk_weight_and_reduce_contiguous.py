from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.topk_weight_and_reduce_contiguous.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reduce contiguous top-k expert outputs into token outputs."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
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
        num_tokens, num_topk = topk_ids.shape
        hidden_dim = fused_expert_output.shape[-1]
        if fused_expert_output.ndim == 2:
            fused_expert_output = fused_expert_output.view(num_tokens, num_topk, hidden_dim)

        values = fused_expert_output.to(self.compute_dtype)
        if not self.apply_router_weight_on_input:
            values = values * topk_weights.to(self.compute_dtype).unsqueeze(-1)
        return values.sum(dim=1).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    topk = int(shape["topk"])
    hidden_dim = int(shape["hidden_dim"])
    dtype = _resolve_float_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    fused_expert_output = torch.randn(
        (num_tokens, topk, hidden_dim),
        dtype=torch.float32,
    ).to(dtype)
    topk_weights = _normalize_prob_rows(
        torch.rand((num_tokens, topk), dtype=torch.float32)
    ).to(dtype)
    topk_ids = torch.randint(
        0,
        max(1, topk * 4),
        (num_tokens, topk),
        dtype=index_dtype,
    )
    return fused_expert_output, topk_weights, topk_ids


def _normalize_prob_rows(weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return weights / denom
