from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.biased_moe_topk_routing.helpers import (
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Select biased top-k experts from router scores for MoE routing."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.topk = int(shape["topk"])
        self.renormalize = bool(shape["renormalize"])
        self.scoring_func = shape["scoring_func"]
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
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gating_output_compute = gating_output.to(self.compute_dtype)
        bias_compute = e_score_correction_bias.to(self.compute_dtype)
        if self.scoring_func == "softmax":
            scores = self._stable_softmax(gating_output_compute, dim=-1)
        else:
            scores = gating_output_compute.sigmoid()

        scores_for_choice = scores + bias_compute.unsqueeze(0)
        topk_ids = self._stable_topk_indices(scores_for_choice, self.topk)
        topk_weights = scores.gather(1, topk_ids)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(self.output_dtype), topk_ids.to(self.index_dtype)

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
    dtype = _resolve_float_dtype(shape["dtype"])
    gating_output = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
    ).to(dtype)
    e_score_correction_bias = torch.randn(
        (num_experts,),
        dtype=torch.float32,
    ).to(dtype)
    return gating_output, e_score_correction_bias
