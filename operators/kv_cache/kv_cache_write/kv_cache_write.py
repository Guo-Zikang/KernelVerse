from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.kv_cache_write.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Write current key/value states into a preallocated KV cache."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )
        self.position_mode = shape.get("position_mode", "shared")

    def forward(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache_compute = key_cache.to(self.compute_dtype).clone()
        value_cache_compute = value_cache.to(self.compute_dtype).clone()
        key_compute = key.to(self.compute_dtype)
        value_compute = value.to(self.compute_dtype)
        positions = positions.long()

        if self.position_mode == "shared":
            key_cache_compute[:, :, positions, :] = key_compute
            value_cache_compute[:, :, positions, :] = value_compute
        else:
            batch_size, num_kv_heads, write_seq_len, _ = key_compute.shape
            batch_indices = torch.arange(
                batch_size,
                device=key_compute.device,
            ).view(batch_size, 1, 1)
            head_indices = torch.arange(
                num_kv_heads,
                device=key_compute.device,
            ).view(1, num_kv_heads, 1)
            pos_indices = positions.view(batch_size, 1, write_seq_len)
            key_cache_compute[batch_indices, head_indices, pos_indices, :] = key_compute
            value_cache_compute[batch_indices, head_indices, pos_indices, :] = value_compute

        return (
            key_cache_compute.to(self.output_dtype),
            value_cache_compute.to(self.output_dtype),
        )


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    num_kv_heads = int(shape["num_kv_heads"])
    max_seq_len = int(shape["max_seq_len"])
    write_seq_len = int(shape["write_seq_len"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    position_mode = shape.get("position_mode", "shared")

    cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
    write_shape = (batch_size, num_kv_heads, write_seq_len, head_dim)
    key_cache = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    value_cache = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    key = torch.randn(write_shape, dtype=torch.float32).to(dtype)
    value = torch.randn(write_shape, dtype=torch.float32).to(dtype)

    if position_mode == "shared":
        positions = _make_positions(max_seq_len, write_seq_len)
    else:
        positions = torch.stack(
            [
                _make_positions(max_seq_len, write_seq_len, batch_idx)
                for batch_idx in range(batch_size)
            ],
            dim=0,
        )

    return key_cache, value_cache, key, value, positions


def _make_positions(
    max_seq_len: int,
    write_seq_len: int,
    offset_seed: int = 0,
) -> torch.Tensor:
    max_start = max_seq_len - write_seq_len
    start = 0 if max_start == 0 else (offset_seed * write_seq_len) % (max_start + 1)
    return torch.arange(start, start + write_seq_len, dtype=torch.long)
