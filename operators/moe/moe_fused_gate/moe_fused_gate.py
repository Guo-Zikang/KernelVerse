from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.moe_fused_gate.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference MoE fused gate with grouped expert routing."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_expert_group = int(shape["num_expert_group"])
        self.topk_group = int(shape["topk_group"])
        self.topk = int(shape["topk"])
        self.num_fused_shared_experts = int(shape["num_fused_shared_experts"])
        self.renormalize = True
        self.routed_scaling_factor = 2.5
        self.apply_routed_scaling_factor_on_output = False
        self.output_dtype = _resolve_float_dtype(shape["dtype"])
        self.index_dtype = _resolve_index_dtype(shape["index_dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        scores: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores_sigmoid = scores.to(self.compute_dtype).sigmoid()
        num_token, num_experts = scores_sigmoid.shape
        experts_per_group = num_experts // self.num_expert_group

        scores_for_choice = scores_sigmoid + bias.to(self.compute_dtype).unsqueeze(0)
        group_scores = scores_for_choice.view(
            num_token,
            self.num_expert_group,
            experts_per_group,
        )
        group_top2_ids = torch.argsort(
            group_scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., :2]
        group_scores = torch.gather(group_scores, -1, group_top2_ids).sum(dim=-1)
        group_idx = self._stable_topk_indices(group_scores, self.topk_group)
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(-1).expand(
            num_token,
            self.num_expert_group,
            experts_per_group,
        ).reshape(num_token, num_experts)
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_excluding_shared = self.topk - self.num_fused_shared_experts
        routed_topk_ids = self._stable_topk_indices(masked_scores, topk_excluding_shared)
        routed_topk_weights = scores_sigmoid.gather(1, routed_topk_ids)

        if self.num_fused_shared_experts > 0:
            topk_ids = torch.empty(
                (num_token, self.topk),
                dtype=routed_topk_ids.dtype,
                device=routed_topk_ids.device,
            )
            topk_weights = torch.empty(
                (num_token, self.topk),
                dtype=routed_topk_weights.dtype,
                device=routed_topk_weights.device,
            )
            topk_ids[:, :topk_excluding_shared] = routed_topk_ids
            topk_weights[:, :topk_excluding_shared] = routed_topk_weights
            routed_sum = routed_topk_weights.sum(dim=-1, keepdim=True)
            for index in range(self.num_fused_shared_experts):
                topk_ids[:, topk_excluding_shared + index] = num_experts + index
                topk_weights[:, topk_excluding_shared + index] = (
                    routed_sum[:, 0] / self.routed_scaling_factor
                )
        else:
            topk_ids = routed_topk_ids
            topk_weights = routed_topk_weights

        if self.renormalize:
            if self.num_fused_shared_experts > 0:
                weight_sum = topk_weights[:, :topk_excluding_shared].sum(
                    dim=-1,
                    keepdim=True,
                )
            else:
                weight_sum = topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights / weight_sum
            if self.apply_routed_scaling_factor_on_output:
                topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(self.output_dtype), topk_ids.to(self.index_dtype)

    def _stable_topk_indices(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    num_experts = int(shape["num_experts"])
    dtype = _resolve_float_dtype(shape["dtype"])
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32).to(dtype)
    bias = torch.rand((num_experts,), dtype=torch.float32).to(dtype)
    return scores, bias
