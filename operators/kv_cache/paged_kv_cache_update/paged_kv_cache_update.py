from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.paged_kv_cache_update.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Update paged KV cache slots with new token states."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(
        self,
        to_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        out = kv_cache.clone()
        flat_cache = out.view(-1, *out.shape[2:])
        valid = slot_mapping >= 0
        flat_cache[slot_mapping[valid].long()] = to_cache[valid].to(flat_cache.dtype)
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    num_blocks = int(shape["num_blocks"])
    block_size = int(shape["block_size"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    to_cache = torch.randn(
        (num_tokens, num_heads, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    kv_cache = torch.zeros(
        (num_blocks, block_size, num_heads, head_dim),
        dtype=dtype,
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size,
        dtype=torch.int64,
    )[:num_tokens]
    if bool(shape["has_invalid_slots"]):
        slot_mapping[::7] = -1
    return to_cache, kv_cache, slot_mapping.to(index_dtype)
