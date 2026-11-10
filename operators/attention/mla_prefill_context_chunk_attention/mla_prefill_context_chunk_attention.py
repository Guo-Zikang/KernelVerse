from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.mla_prefill_context_chunk_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


def _make_cu_seqlens(seq_lens, device: torch.device) -> torch.Tensor:
    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + int(seq_len))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


class Model(nn.Module):
    """Run non-causal variable-length MLA prefill attention over context chunks."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.qk_head_dim = int(shape["qk_head_dim"])
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
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        value_compute = value.to(self.compute_dtype)
        outputs = []
        lses = []
        num_sequences = cu_seqlens_q.numel() - 1
        scale = self.qk_head_dim ** -0.5

        for seq_idx in range(num_sequences):
            q_start = int(cu_seqlens_q[seq_idx].item())
            q_end = int(cu_seqlens_q[seq_idx + 1].item())
            k_start = int(cu_seqlens_k[seq_idx].item())
            k_end = int(cu_seqlens_k[seq_idx + 1].item())
            q_slice = query_compute[q_start:q_end].permute(1, 0, 2).unsqueeze(0)
            k_slice = key_compute[k_start:k_end].permute(1, 0, 2).unsqueeze(0)
            v_slice = value_compute[k_start:k_end].permute(1, 0, 2).unsqueeze(0)

            scores = torch.einsum("bhqd,bhkd->bhqk", q_slice, k_slice) * scale
            probs = self._stable_softmax(scores)
            out = torch.einsum("bhqk,bhkd->bhqd", probs, v_slice)
            lse = self._stable_logsumexp(scores.to(torch.float32)).squeeze(-1)
            outputs.append(out.squeeze(0).permute(1, 0, 2))
            lses.append(lse.squeeze(0).permute(1, 0))

        return torch.cat(outputs, dim=0).to(self.output_dtype), torch.cat(lses, dim=0)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    q_seq_lens = [int(seq_len) for seq_len in shape["q_seq_lens"]]
    kv_seq_lens = [int(seq_len) for seq_len in shape["kv_seq_lens"]]
    total_q = sum(q_seq_lens)
    total_k = sum(kv_seq_lens)
    num_heads = int(shape["num_heads"])
    qk_head_dim = int(shape["qk_head_dim"])
    v_head_dim = int(shape["v_head_dim"])
    query = torch.randn((total_q, num_heads, qk_head_dim), dtype=torch.float32).to(dtype)
    key = torch.randn((total_k, num_heads, qk_head_dim), dtype=torch.float32).to(dtype)
    value = torch.randn((total_k, num_heads, v_head_dim), dtype=torch.float32).to(dtype)
    cu_q = _make_cu_seqlens(q_seq_lens, query.device)
    cu_k = _make_cu_seqlens(kv_seq_lens, query.device)
    return query, key, value, cu_q, cu_k
