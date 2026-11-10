from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.grouped_topk_routing.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Selects top-k MoE experts after grouped routing score filtering."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.topk = int(shape["topk"])
        self.num_expert_group = int(shape["num_expert_group"])
        self.topk_group = int(shape["topk_group"])
        self.renormalize = bool(shape["renormalize"])
        self.scoring_func = shape["scoring_func"]
        self.routed_scaling_factor = float(shape["routed_scaling_factor"])
        self.has_correction_bias = bool(shape["has_correction_bias"])
        self.weight_dtype = _resolve_float_dtype(shape["weight_dtype"])
        self.index_dtype = _resolve_index_dtype(shape["index_dtype"])

    def forward(
        self,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scoring_func == "softmax":
            scores = self._stable_softmax(gating_output.float(), dim=-1)
        else:
            scores = gating_output.float().sigmoid()

        num_tokens = scores.size(0)
        experts_per_group = scores.size(-1) // self.num_expert_group
        if self.has_correction_bias:
            original_scores = scores
            biased_scores = scores + e_score_correction_bias.float().unsqueeze(0)
            top_m = min(2, experts_per_group)
            grouped_scores = biased_scores.view(
                num_tokens,
                self.num_expert_group,
                experts_per_group,
            )
            group_top_m_ids = torch.argsort(
                grouped_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[..., :top_m]
            group_scores = torch.gather(grouped_scores, -1, group_top_m_ids).sum(dim=-1)
        else:
            original_scores = None
            biased_scores = scores
            group_scores = biased_scores.view(
                num_tokens,
                self.num_expert_group,
                experts_per_group,
            ).max(dim=-1).values

        group_idx = self._stable_topk_indices(group_scores, self.topk_group)
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(
            num_tokens,
            self.num_expert_group,
            experts_per_group,
        ).reshape(num_tokens, -1)
        masked_scores = biased_scores.masked_fill(~score_mask.bool(), float("-inf"))

        if original_scores is not None:
            topk_ids = self._stable_topk_indices(masked_scores, self.topk)
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_ids = self._stable_topk_indices(masked_scores, self.topk)
            topk_weights = masked_scores.gather(1, topk_ids)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(self.weight_dtype), topk_ids.to(self.index_dtype)

    def _stable_softmax(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        shifted = scores - scores.max(dim=dim, keepdim=True).values
        exp_scores = torch.exp(shifted)
        return exp_scores / exp_scores.sum(dim=dim, keepdim=True)

    def _stable_topk_indices(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    num_experts = int(shape["num_experts"])
    input_dtype = _resolve_float_dtype(shape["input_dtype"])
    gating_output = torch.randn((num_tokens, num_experts), dtype=torch.float32).to(input_dtype)
    e_score_correction_bias = torch.randn((num_experts,), dtype=torch.float32).to(input_dtype)
    return gating_output, e_score_correction_bias
