from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.concat_and_cache_mla_rope.helpers import (
    _resolve_dtype,
    validate_shape,
)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    work_x = x.to(compute_dtype)
    work_cos = cos.to(compute_dtype)
    work_sin = sin.to(compute_dtype)
    while work_cos.ndim < work_x.ndim:
        work_cos = work_cos.unsqueeze(-2)
        work_sin = work_sin.unsqueeze(-2)

    if is_neox:
        x1, x2 = torch.chunk(work_x, 2, dim=-1)
        rotated = torch.cat(
            (x1 * work_cos - x2 * work_sin, x2 * work_cos + x1 * work_sin),
            dim=-1,
        )
    else:
        x1 = work_x[..., ::2]
        x2 = work_x[..., 1::2]
        rotated = torch.stack(
            (x1 * work_cos - x2 * work_sin, x2 * work_cos + x1 * work_sin),
            dim=-1,
        ).flatten(-2)
    return rotated.to(output_dtype)


class Model(nn.Module):
    """Applies RoPE to MLA query/key features and writes fused KV cache entries."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.is_neox = bool(shape.get("is_neox", True))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotary_dim = q_pe.shape[-1]
        half = rotary_dim // 2
        pos = positions.to(torch.long)
        cos = cos_sin_cache[pos, :half]
        sin = cos_sin_cache[pos, half:]

        q_rot = _apply_rotary(
            q_pe,
            cos,
            sin,
            self.is_neox,
            self.compute_dtype,
            self.output_dtype,
        )
        k_rot = _apply_rotary(
            k_pe,
            cos,
            sin,
            self.is_neox,
            self.compute_dtype,
            self.output_dtype,
        )

        to_cache = torch.cat(
            [kv_c.to(self.compute_dtype), k_rot.to(self.compute_dtype)],
            dim=-1,
        ).to(kv_cache.dtype)
        cached = kv_cache.clone()
        flat_cache = cached.view(-1, *cached.shape[2:])
        valid = slot_mapping >= 0
        flat_cache[slot_mapping[valid].to(torch.long)] = to_cache[valid].to(flat_cache.dtype)
        return q_rot, k_rot, cached


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = int(shape["num_tokens"])
    num_blocks = int(shape["num_blocks"])
    block_size = int(shape["block_size"])
    latent_dim = int(shape["latent_dim"])
    rope_dim = int(shape["rope_dim"])
    max_position = int(shape["max_position"])

    positions = torch.arange(num_tokens, dtype=torch.int64) % max_position
    q_pe = torch.randn((num_tokens, rope_dim), dtype=torch.float32).to(dtype)
    k_pe = torch.randn((num_tokens, rope_dim), dtype=torch.float32).to(dtype)
    kv_c = torch.randn((num_tokens, latent_dim), dtype=torch.float32).to(dtype)
    cos_sin_cache = torch.randn((max_position, rope_dim), dtype=torch.float32)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64) % (num_blocks * block_size)
    kv_cache = torch.zeros((num_blocks, block_size, latent_dim + rope_dim), dtype=dtype)
    return positions, q_pe, k_pe, kv_c, cos_sin_cache, slot_mapping, kv_cache
