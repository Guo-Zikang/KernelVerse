from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_topk_routing.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Select top-k experts from router scores for MoE routing."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.topk = int(shape["topk"])
        self.renormalize = bool(shape["renormalize"])
        self.scoring_func = shape["scoring_func"]
        self.weight_dtype = _resolve_float_dtype(shape["weight_dtype"])
        self.index_dtype = _resolve_index_dtype(shape["index_dtype"])

    def forward(
        self,
        gating_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scoring_func == "softmax":
            scores = self._stable_softmax(gating_output.float(), dim=-1)
        else:
            scores = gating_output.float().sigmoid()

        topk_ids = self._stable_topk_indices(scores, self.topk)
        topk_weights = scores.gather(1, topk_ids)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        num_tokens = gating_output.size(0)
        token_expert_indices = torch.arange(
            num_tokens * self.topk,
            dtype=torch.int32,
            device=gating_output.device,
        ).reshape(num_tokens, self.topk)
        return (
            topk_weights.to(self.weight_dtype),
            topk_ids.to(self.index_dtype),
            token_expert_indices.to(self.index_dtype),
        )

    def _stable_softmax(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        shifted = scores - scores.max(dim=dim, keepdim=True).values
        exp_scores = torch.exp(shifted)
        return exp_scores / exp_scores.sum(dim=dim, keepdim=True)

    def _stable_topk_indices(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    num_experts = int(shape["num_experts"])
    input_dtype = _resolve_float_dtype(shape["input_dtype"])
    gating_output = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
    ).to(input_dtype)
    return (gating_output,)
