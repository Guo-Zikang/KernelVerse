from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.grouped_moe_mm.helpers import (
    _resolve_dtype,
    validate_shape,
)


def _expand_rows(scale: torch.Tensor, rows: int) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(rows, 1)
    if scale.ndim == 1 and scale.shape[0] == rows:
        return scale.reshape(rows, 1)
    if scale.ndim == 2 and scale.shape == (rows, 1):
        return scale
    return scale.reshape(rows, -1)


def _expand_cols(scale: torch.Tensor, cols: int) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(1, cols)
    if scale.ndim == 1 and scale.shape[0] == cols:
        return scale.reshape(1, cols)
    if scale.ndim == 2 and scale.shape == (1, cols):
        return scale
    return scale.reshape(-1, cols)


class Model(nn.Module):
    """Runs per-expert MoE matrix multiplications with optional scale factors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.per_act_token = bool(shape["per_act_token"])
        self.per_out_ch = bool(shape["per_out_ch"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        a_tensors: torch.Tensor,
        b_tensors: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = b_tensors.shape[0]
        outputs = []
        a_compute = a_tensors.to(self.compute_dtype)
        b_compute = b_tensors.to(self.compute_dtype)
        a_scale_compute = a_scales.to(self.compute_dtype)
        b_scale_compute = b_scales.to(self.compute_dtype)

        for expert_idx in range(num_experts):
            start = int(expert_offsets[expert_idx].item())
            end = int(expert_offsets[expert_idx + 1].item())
            if end <= start:
                continue

            a_chunk = a_compute[start:end]
            weight = b_compute[expert_idx]

            if self.per_act_token:
                a_scale = _expand_rows(a_scale_compute[start:end], end - start)
            else:
                a_scale = _expand_rows(a_scale_compute.reshape(-1)[:1], end - start)

            if self.per_out_ch:
                b_scale = _expand_cols(b_scale_compute[expert_idx], weight.shape[1])
            else:
                scale_idx = expert_idx if b_scale_compute.numel() > 1 else 0
                scalar_scale = b_scale_compute.reshape(-1)[scale_idx : scale_idx + 1]
                b_scale = _expand_cols(scalar_scale, weight.shape[1])

            outputs.append((a_chunk * a_scale) @ (weight * b_scale))

        if not outputs:
            return torch.empty(
                (0, b_tensors.shape[-1]),
                dtype=self.output_dtype,
                device=a_tensors.device,
            )
        return torch.cat(outputs, dim=0).to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_experts = int(shape["num_experts"])
    tokens_per_expert = int(shape["tokens_per_expert"])
    k_dim = int(shape["K"])
    n_dim = int(shape["N"])
    total_tokens = num_experts * tokens_per_expert

    a_tensors = torch.randn((total_tokens, k_dim), dtype=torch.float32).to(dtype)
    b_tensors = torch.randn((num_experts, k_dim, n_dim), dtype=torch.float32).to(dtype)

    if bool(shape["per_act_token"]):
        a_scales = (torch.rand((total_tokens,), dtype=torch.float32) * 0.02 + 0.001).to(dtype)
    else:
        a_scales = (torch.rand((1,), dtype=torch.float32) * 0.02 + 0.001).to(dtype)

    if bool(shape["per_out_ch"]):
        b_scales = (torch.rand((num_experts, n_dim), dtype=torch.float32) * 0.02 + 0.001).to(dtype)
    else:
        b_scales = (torch.rand((num_experts,), dtype=torch.float32) * 0.02 + 0.001).to(dtype)

    counts = torch.full((num_experts,), tokens_per_expert, dtype=torch.int32)
    expert_offsets = torch.zeros((num_experts + 1,), dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(counts, dim=0)
    return a_tensors, b_tensors, a_scales, b_scales, expert_offsets
