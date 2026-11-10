from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.transfer_mla_kv_per_layer.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Transfer compressed MLA KV-cache pages for one layer."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = dst.clone()
        out[dst_indices.long()] = src[src_indices.long()]
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_pages = int(shape["num_pages"])
    num_items = int(shape["num_items"])
    item_elems = int(shape["item_elems"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    cache_shape = (num_pages, item_elems)
    src = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    dst = torch.zeros(cache_shape, dtype=dtype)
    src_indices = torch.randperm(num_pages, dtype=torch.int64)[:num_items].to(index_dtype)
    dst_indices = torch.randperm(num_pages, dtype=torch.int64)[:num_items].to(index_dtype)
    return src, dst, src_indices, dst_indices
