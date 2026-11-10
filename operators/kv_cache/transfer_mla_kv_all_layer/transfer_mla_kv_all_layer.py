from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.transfer_mla_kv_all_layer.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Transfer compressed MLA KV-cache pages across all layers."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(
        self,
        src_layers: torch.Tensor,
        dst_layers: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = dst_layers.clone()
        out[:, dst_indices.long()] = src_layers[:, src_indices.long()]
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_layers = int(shape["num_layers"])
    num_pages = int(shape["num_pages"])
    num_items = int(shape["num_items"])
    item_elems = int(shape["item_elems"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    cache_shape = (num_layers, num_pages, item_elems)
    src_layers = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    dst_layers = torch.zeros(cache_shape, dtype=dtype)
    src_indices = torch.randperm(num_pages, dtype=torch.int64)[:num_items].to(index_dtype)
    dst_indices = torch.randperm(num_pages, dtype=torch.int64)[:num_items].to(index_dtype)
    return src_layers, dst_layers, src_indices, dst_indices
