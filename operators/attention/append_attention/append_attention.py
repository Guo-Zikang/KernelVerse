from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.append_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference fused append-KV-cache and decode attention."""

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
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_updated = k_cache.to(self.compute_dtype).clone()
        v_updated = v_cache.to(self.compute_dtype).clone()
        batch_indices = torch.arange(q.shape[0], device=q.device)
        append_positions = seq_lens.to(q.device).long()
        k_updated[batch_indices, append_positions, :, :] = new_k.to(self.compute_dtype)
        v_updated[batch_indices, append_positions, :, :] = new_v.to(self.compute_dtype)

        q_compute = q.to(self.compute_dtype)
        q_per_kv = self.num_q_heads // self.num_kv_heads
        kv_head_indices = torch.arange(self.num_q_heads, device=q.device) // q_per_kv
        key = k_updated[:, :, kv_head_indices, :].permute(0, 2, 1, 3)
        value = v_updated[:, :, kv_head_indices, :].permute(0, 2, 1, 3)

        scores = (key * q_compute.unsqueeze(2)).sum(dim=-1) * self.scale
        positions = torch.arange(k_cache.shape[1], device=q.device).view(1, 1, -1)
        valid_lens = append_positions.view(-1, 1, 1) + 1
        scores = scores.masked_fill(positions >= valid_lens, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = (weights.unsqueeze(-1) * value).sum(dim=2)
        return (
            output.to(self.output_dtype),
            k_updated.to(self.output_dtype),
            v_updated.to(self.output_dtype),
        )


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    max_seq_len = int(shape["max_seq_len"])
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])

    q = torch.randn((batch_size, num_q_heads, head_dim), dtype=torch.float32).to(dtype)
    new_k = torch.randn((batch_size, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    new_v = torch.randn((batch_size, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    k_cache = torch.randn(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    v_cache = torch.randn(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    seq_lens = _make_seq_lens(batch_size, max_seq_len, str(shape["seq_len_pattern"]))
    return q, new_k, new_v, k_cache, v_cache, seq_lens


def _make_seq_lens(batch_size: int, max_seq_len: int, pattern: str) -> torch.Tensor:
    max_existing_len = max_seq_len - 1
    if pattern == "empty":
        return torch.zeros((batch_size,), dtype=torch.int64)
    if pattern == "uniform_short":
        value = max(1, max_seq_len // 8)
        return torch.full((batch_size,), min(value, max_existing_len), dtype=torch.int64)
    if pattern == "uniform_medium":
        value = max(1, max_seq_len // 2)
        return torch.full((batch_size,), min(value, max_existing_len), dtype=torch.int64)
    if pattern == "uniform_long":
        return torch.full((batch_size,), max_existing_len, dtype=torch.int64)
    if pattern == "ragged":
        low = max(1, max_seq_len // 4)
        return torch.randint(low, max_seq_len, (batch_size,), dtype=torch.int64)
    if pattern == "highly_ragged":
        if batch_size == 1:
            return torch.tensor([max_existing_len], dtype=torch.int64)
        base = torch.linspace(0, max_existing_len, steps=batch_size)
        return base.round().clamp(min=0, max=max_existing_len).to(torch.int64)
    raise ValueError(f"Unsupported seq_len_pattern: {pattern}")
