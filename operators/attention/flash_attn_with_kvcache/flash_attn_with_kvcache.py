from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.flash_attn_with_kvcache.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Scaled dot-product attention from query tokens over a KV cache."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_dim = int(shape["head_dim"])
        self.softmax_scale = float(shape.get("softmax_scale", self.head_dim ** -0.5))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_compute = q.to(self.compute_dtype)
        k_compute = k_cache.to(self.compute_dtype)
        v_compute = v_cache.to(self.compute_dtype)
        outputs = []
        lse_parts = []

        for batch_idx in range(q_compute.shape[0]):
            valid_k = int(cache_seqlens[batch_idx].item())
            q_slice = q_compute[batch_idx]
            k_slice = k_compute[batch_idx, :valid_k]
            v_slice = v_compute[batch_idx, :valid_k]

            scores = torch.einsum("qhd,khd->hqk", q_slice, k_slice)
            scores = scores * self.softmax_scale
            probs = F.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", probs, v_slice))
            lse_parts.append(torch.logsumexp(scores.to(torch.float32), dim=-1))

        output = torch.stack(outputs, dim=0).to(self.output_dtype)
        softmax_lse = torch.stack(lse_parts, dim=0).to(torch.float32)
        return output, softmax_lse


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len_q = int(shape["seq_len_q"])
    seq_len_k = int(shape["seq_len_k"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    k_cache = torch.randn(
        (batch_size, seq_len_k, num_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    v_cache = torch.randn(
        (batch_size, seq_len_k, num_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    cache_seqlens = _build_cache_seqlens(
        batch_size=batch_size,
        seq_len_k=seq_len_k,
        cache_pattern=str(shape.get("cache_pattern", "full")),
    )
    return q, k_cache, v_cache, cache_seqlens


def _build_cache_seqlens(
    *,
    batch_size: int,
    seq_len_k: int,
    cache_pattern: str,
) -> torch.Tensor:
    if cache_pattern == "ragged" and batch_size > 1:
        lengths = []
        min_len = max(1, seq_len_k // 2)
        span = max(1, seq_len_k - min_len)
        for batch_idx in range(batch_size):
            offset = (batch_idx * span) // max(1, batch_size - 1)
            lengths.append(seq_len_k - offset)
        return torch.tensor(lengths, dtype=torch.int32)
    return torch.full((batch_size,), seq_len_k, dtype=torch.int32)
