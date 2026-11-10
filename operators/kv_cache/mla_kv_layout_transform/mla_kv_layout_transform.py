from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.mla_kv_layout_transform.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Transform compressed MLA KV-cache layout between paged and layer-first forms."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.layout_mode = shape["layout_mode"]
        self.layer_id = int(shape.get("layer_id", 0))
        self.page_size = int(shape["page_size"])
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = dst.clone()

        if self.layout_mode == "per_layer_pf_lf":
            for index in range(src_indices.numel()):
                src_idx = int(src_indices[index].item())
                dst_idx = int(dst_indices[index].item())
                page = src_idx // self.page_size
                offset = src_idx % self.page_size
                out[dst_idx] = src[page, self.layer_id, offset]
        else:
            for layer in range(src.shape[0]):
                for index in range(src_indices.numel()):
                    src_idx = int(src_indices[index].item())
                    dst_idx = int(dst_indices[index].item())
                    page = dst_idx // self.page_size
                    offset = dst_idx % self.page_size
                    out[page, layer, offset] = src[layer, src_idx]

        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    layout_mode = shape["layout_mode"]
    num_layers = int(shape["num_layers"])
    num_pages = int(shape["num_pages"])
    page_size = int(shape["page_size"])
    num_items = int(shape["num_items"])
    item_elems = int(shape["item_elems"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    if layout_mode == "per_layer_pf_lf":
        src_shape = (num_pages, num_layers, page_size, item_elems)
        dst_shape = (num_pages * page_size, item_elems)
    else:
        src_shape = (num_layers, num_pages * page_size, item_elems)
        dst_shape = (num_pages, num_layers, page_size, item_elems)

    src = torch.randn(src_shape, dtype=torch.float32).to(dtype)
    dst = torch.zeros(dst_shape, dtype=dtype)
    src_indices, dst_indices = _make_indices(num_pages, page_size, num_items, index_dtype)
    return src, dst, src_indices, dst_indices


def _make_indices(
    num_pages: int,
    page_size: int,
    num_items: int,
    index_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total = num_pages * page_size
    src_indices = torch.randperm(total, dtype=torch.int64)[:num_items].to(index_dtype)
    dst_indices = torch.randperm(total, dtype=torch.int64)[:num_items].to(index_dtype)
    return src_indices, dst_indices
