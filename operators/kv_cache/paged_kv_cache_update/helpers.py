"""Helper utilities for paged_kv_cache_update."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "num_blocks",
    "block_size",
    "num_heads",
    "head_dim",
    "dtype",
    "index_dtype",
    "has_invalid_slots",
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

    for key in ("num_tokens", "num_blocks", "block_size", "num_heads", "head_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if shape["num_tokens"] > shape["num_blocks"] * shape["block_size"]:
        raise ValueError("num_tokens must be less than or equal to total cache slots.")
    if not isinstance(shape["has_invalid_slots"], bool):
        raise ValueError("has_invalid_slots must be a boolean.")

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
