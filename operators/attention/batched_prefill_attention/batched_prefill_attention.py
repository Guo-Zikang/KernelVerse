from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.batched_prefill_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference packed batched prefill attention for LLM serving."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_q_heads = int(shape["num_q_heads"])
        self.num_kv_heads = int(shape["num_kv_heads"])
        self.head_dim = int(shape["head_dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        self.softmax_scale = float(shape.get("softmax_scale", self.head_dim ** -0.5))
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
    ) -> torch.Tensor:
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)

        q_per_kv = self.num_q_heads // self.num_kv_heads
        kv_head_indices = torch.arange(self.num_q_heads, device=q.device) // q_per_kv
        outputs = []

        for batch_idx in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            kv_start = int(kv_indptr[batch_idx].item())
            kv_end = int(kv_indptr[batch_idx + 1].item())

            q_slice = q_compute[q_start:q_end]
            k_slice = k_compute[kv_start:kv_end, kv_head_indices, :]
            v_slice = v_compute[kv_start:kv_end, kv_head_indices, :]
            q_len = q_slice.shape[0]
            kv_len = k_slice.shape[0]

            scores = torch.einsum("qhd,khd->hqk", q_slice, k_slice)
            scores = scores * self.softmax_scale
            scores = self._apply_causal_mask(scores, q_len, kv_len)
            probs = F.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", probs, v_slice))

        return torch.cat(outputs, dim=0).to(self.output_dtype)

    @staticmethod
    def _apply_causal_mask(scores: torch.Tensor, q_len: int, kv_len: int) -> torch.Tensor:
        query_positions = torch.arange(q_len, device=scores.device).unsqueeze(1)
        key_positions = torch.arange(kv_len, device=scores.device).unsqueeze(0)
        absolute_query_positions = query_positions + kv_len - q_len
        mask = key_positions > absolute_query_positions
        return scores.masked_fill(mask.unsqueeze(0), float("-inf"))


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    q_lens = [int(seq_len) for seq_len in shape["q_lens"]]
    kv_lens = [int(seq_len) for seq_len in shape["kv_lens"]]
    total_q_tokens = sum(q_lens)
    total_kv_tokens = sum(kv_lens)
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])

    q = torch.randn((total_q_tokens, num_q_heads, head_dim), dtype=torch.float32).to(dtype)
    k = torch.randn((total_kv_tokens, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    v = torch.randn((total_kv_tokens, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    qo_indptr = _make_indptr(q_lens)
    kv_indptr = _make_indptr(kv_lens)
    return q, k, v, qo_indptr, kv_indptr


def _make_indptr(seq_lens: list[int]) -> torch.Tensor:
    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + int(seq_len))
    return torch.tensor(offsets, dtype=torch.int32)
