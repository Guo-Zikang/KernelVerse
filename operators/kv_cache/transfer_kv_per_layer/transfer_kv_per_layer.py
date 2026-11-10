from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.transfer_kv_per_layer.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Transfer K/V cache pages for one layer."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(
        self,
        src_k: torch.Tensor,
        dst_k: torch.Tensor,
        src_v: torch.Tensor,
        dst_v: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_k = dst_k.clone()
        out_v = dst_v.clone()
        out_k[dst_indices.long()] = src_k[src_indices.long()]
        out_v[dst_indices.long()] = src_v[src_indices.long()]
        return out_k.to(self.output_dtype), out_v.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_pages = int(shape["num_pages"])
    num_items = int(shape["num_items"])
    item_elems = int(shape["item_elems"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    cache_shape = (num_pages, item_elems)
    src_k = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    src_v = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    dst_k = torch.zeros(cache_shape, dtype=dtype)
    dst_v = torch.zeros(cache_shape, dtype=dtype)
    src_indices = torch.randperm(num_pages, dtype=torch.int64)[:num_items].to(index_dtype)
    dst_indices = torch.randperm(num_pages, dtype=torch.int64)[:num_items].to(index_dtype)
    return src_k, dst_k, src_v, dst_v, src_indices, dst_indices
