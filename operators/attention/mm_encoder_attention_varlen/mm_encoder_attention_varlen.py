from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.mm_encoder_attention_varlen.helpers import (
    _resolve_dtype,
    validate_shape,
)


def _make_cu_seqlens(seq_lens, device: torch.device) -> torch.Tensor:
    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + int(seq_len))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


class Model(nn.Module):
    """Run non-causal variable-length multimodal encoder attention."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_dim = int(shape["head_dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def _stable_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        exp_sum = exp_scores.sum(dim=-1, keepdim=True)
        return exp_scores / exp_sum

    def _stable_logsumexp(self, scores: torch.Tensor) -> torch.Tensor:
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        exp_sum = exp_scores.sum(dim=-1, keepdim=True)
        return torch.log(exp_sum) + score_max

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        value_compute = value.to(self.compute_dtype)
        outputs = []
        lses = []
        num_sequences = cu_seqlens.numel() - 1
        scale = self.head_dim ** -0.5

        for seq_idx in range(num_sequences):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            q_slice = query_compute[start:end].permute(1, 0, 2).unsqueeze(0)
            k_slice = key_compute[start:end].permute(1, 0, 2).unsqueeze(0)
            v_slice = value_compute[start:end].permute(1, 0, 2).unsqueeze(0)

            scores = torch.einsum("bhqd,bhkd->bhqk", q_slice, k_slice) * scale
            probs = self._stable_softmax(scores)
            out = torch.einsum("bhqk,bhkd->bhqd", probs, v_slice)
            lse = self._stable_logsumexp(scores.to(torch.float32)).squeeze(-1)
            outputs.append(out.squeeze(0).permute(1, 0, 2))
            lses.append(lse.squeeze(0).permute(1, 0))

        return torch.cat(outputs, dim=0).to(self.output_dtype), torch.cat(lses, dim=0)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    seq_lens = [int(seq_len) for seq_len in shape["seq_lens"]]
    total_tokens = sum(seq_lens)
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dims = (total_tokens, num_heads, head_dim)
    q = torch.randn(dims, dtype=torch.float32).to(dtype)
    k = torch.randn(dims, dtype=torch.float32).to(dtype)
    v = torch.randn(dims, dtype=torch.float32).to(dtype)
    cu_seqlens = _make_cu_seqlens(seq_lens, q.device)
    return q, k, v, cu_seqlens
