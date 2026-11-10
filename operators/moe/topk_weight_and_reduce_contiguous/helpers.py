"""Helper utilities for topk_weight_and_reduce_contiguous."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "topk",
    "hidden_dim",
    "dtype",
    "index_dtype",
    "apply_router_weight_on_input",
)

FLOAT_DTYPE_NAME_MAP = {
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

    for key in ("num_tokens", "topk", "hidden_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if not isinstance(shape["apply_router_weight_on_input"], bool):
        raise ValueError("apply_router_weight_on_input must be a boolean.")

    _resolve_float_dtype(shape["dtype"])
    _resolve_index_dtype(shape["index_dtype"])


def _resolve_float_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in FLOAT_DTYPE_NAME_MAP:
        supported = ", ".join(sorted(FLOAT_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported floating dtype: {dtype_name}. Supported dtypes: {supported}")
    return FLOAT_DTYPE_NAME_MAP[dtype_name]


def _resolve_index_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in INDEX_DTYPE_NAME_MAP:
        supported = ", ".join(sorted(INDEX_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported index dtype: {dtype_name}. Supported dtypes: {supported}")
    return INDEX_DTYPE_NAME_MAP[dtype_name]
