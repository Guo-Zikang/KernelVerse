from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.kv_layout_transform.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Transform K/V cache layout between paged and layer-first forms."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.layout_mode = shape["layout_mode"]
        self.layer_id = int(shape.get("layer_id", 0))
        self.page_size = int(shape["page_size"])
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

        if self.layout_mode == "per_layer_pf_lf":
            for index in range(src_indices.numel()):
                src_idx = int(src_indices[index].item())
                dst_idx = int(dst_indices[index].item())
                page = src_idx // self.page_size
                offset = src_idx % self.page_size
                out_k[dst_idx] = src_k[page, self.layer_id, offset]
                out_v[dst_idx] = src_v[page, self.layer_id, offset]
        elif self.layout_mode == "per_layer_ph_lf":
            for index in range(src_indices.numel()):
                src_idx = int(src_indices[index].item())
                dst_idx = int(dst_indices[index].item())
                page = src_idx // self.page_size
                offset = src_idx % self.page_size
                out_k[dst_idx] = src_k[page, :, offset, self.layer_id]
                out_v[dst_idx] = src_v[page, :, offset, self.layer_id]
        elif self.layout_mode == "all_layer_lf_pf":
            for layer in range(src_k.shape[0]):
                for index in range(src_indices.numel()):
                    src_idx = int(src_indices[index].item())
                    dst_idx = int(dst_indices[index].item())
                    page = dst_idx // self.page_size
                    offset = dst_idx % self.page_size
                    out_k[page, layer, offset] = src_k[layer, src_idx]
                    out_v[page, layer, offset] = src_v[layer, src_idx]
        else:
            for layer in range(src_k.shape[0]):
                for index in range(src_indices.numel()):
                    src_idx = int(src_indices[index].item())
                    dst_idx = int(dst_indices[index].item())
                    page = dst_idx // self.page_size
                    offset = dst_idx % self.page_size
                    out_k[page, :, offset, layer] = src_k[layer, src_idx]
                    out_v[page, :, offset, layer] = src_v[layer, src_idx]

        return out_k.to(self.output_dtype), out_v.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    layout_mode = shape["layout_mode"]
    num_layers = int(shape["num_layers"])
    num_pages = int(shape["num_pages"])
    page_size = int(shape["page_size"])
    num_items = int(shape["num_items"])
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])

    if layout_mode == "per_layer_pf_lf":
        item_elems = int(shape["item_elems"])
        src_shape = (num_pages, num_layers, page_size, item_elems)
        dst_shape = (num_pages * page_size, item_elems)
    elif layout_mode == "per_layer_ph_lf":
        head_num = int(shape["head_num"])
        head_elems = int(shape["head_elems"])
        src_shape = (num_pages, head_num, page_size, num_layers, head_elems)
        dst_shape = (num_pages * page_size, head_num, head_elems)
    elif layout_mode == "all_layer_lf_pf":
        item_elems = int(shape["item_elems"])
        src_shape = (num_layers, num_pages * page_size, item_elems)
        dst_shape = (num_pages, num_layers, page_size, item_elems)
    else:
        head_num = int(shape["head_num"])
        head_elems = int(shape["head_elems"])
        src_shape = (num_layers, num_pages * page_size, head_num, head_elems)
        dst_shape = (num_pages, head_num, page_size, num_layers, head_elems)

    src_k = torch.randn(src_shape, dtype=torch.float32).to(dtype)
    src_v = torch.randn(src_shape, dtype=torch.float32).to(dtype)
    dst_k = torch.zeros(dst_shape, dtype=dtype)
    dst_v = torch.zeros(dst_shape, dtype=dtype)
    src_indices, dst_indices = _make_indices(num_pages, page_size, num_items, index_dtype)
    return src_k, dst_k, src_v, dst_v, src_indices, dst_indices


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
