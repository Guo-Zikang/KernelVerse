from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.ragged_kvcache_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference single-token decode attention over packed ragged KV cache."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_q_heads = int(shape["num_q_heads"])
        self.num_kv_heads = int(shape["num_kv_heads"])
        self.head_dim = int(shape["head_dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        self.scale = float(shape.get("softmax_scale", self.head_dim ** -0.5))
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
        kv_indptr: torch.Tensor,
    ) -> torch.Tensor:
        q_compute = q.to(self.compute_dtype)
        k_compute = k_cache.to(self.compute_dtype)
        v_compute = v_cache.to(self.compute_dtype)
        q_per_kv = self.num_q_heads // self.num_kv_heads
        kv_head_indices = torch.arange(self.num_q_heads, device=q.device) // q_per_kv

        outputs = []
        for batch_idx in range(kv_indptr.numel() - 1):
            kv_start = int(kv_indptr[batch_idx].item())
            kv_end = int(kv_indptr[batch_idx + 1].item())
            key = k_compute[kv_start:kv_end, kv_head_indices, :]
            value = v_compute[kv_start:kv_end, kv_head_indices, :]

            scores = (key * q_compute[batch_idx].unsqueeze(0)).sum(dim=-1).transpose(0, 1)
            scores = scores * self.scale
            weights = torch.softmax(scores, dim=-1)
            output = (weights.unsqueeze(-1) * value.permute(1, 0, 2)).sum(dim=1)
            outputs.append(output)

        return torch.stack(outputs, dim=0).to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    kv_lens = [int(seq_len) for seq_len in shape["kv_lens"]]
    batch_size = len(kv_lens)
    total_kv_tokens = sum(kv_lens)
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])

    q = torch.randn((batch_size, num_q_heads, head_dim), dtype=torch.float32).to(dtype)
    k_cache = torch.randn(
        (total_kv_tokens, num_kv_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    v_cache = torch.randn(
        (total_kv_tokens, num_kv_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    kv_indptr = _make_indptr(kv_lens)
    return q, k_cache, v_cache, kv_indptr


def _make_indptr(seq_lens: list[int]) -> torch.Tensor:
    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + int(seq_len))
    return torch.tensor(offsets, dtype=torch.int32)
