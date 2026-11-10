"""Helper utilities for mla_kv_layout_transform."""

from typing import Any, Dict

import torch


LAYOUT_MODES = {
    "per_layer_pf_lf",
    "all_layer_lf_pf",
}

REQUIRED_SHAPE_KEYS = (
    "layout_mode",
    "num_layers",
    "num_pages",
    "page_size",
    "num_items",
    "item_elems",
    "dtype",
    "index_dtype",
)

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}

INDEX_DTYPE_NAME_MAP = {
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    layout_mode = shape["layout_mode"]
    if layout_mode not in LAYOUT_MODES:
        supported = ", ".join(sorted(LAYOUT_MODES))
        raise ValueError(f"layout_mode must be one of: {supported}.")

    for key in ("num_layers", "num_pages", "page_size", "num_items", "item_elems"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if shape["num_items"] > shape["num_pages"] * shape["page_size"]:
        raise ValueError("num_items exceeds total page capacity.")

    if layout_mode == "per_layer_pf_lf":
        layer_id = shape.get("layer_id", 0)
        if not isinstance(layer_id, int) or layer_id < 0 or layer_id >= shape["num_layers"]:
            raise ValueError("layer_id must be in [0, num_layers).")

    _resolve_dtype(shape["dtype"])
    _resolve_index_dtype(shape["index_dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]


def _resolve_index_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in INDEX_DTYPE_NAME_MAP:
        supported = ", ".join(sorted(INDEX_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported index dtype: {dtype_name}. Supported dtypes: {supported}")
    return INDEX_DTYPE_NAME_MAP[dtype_name]
