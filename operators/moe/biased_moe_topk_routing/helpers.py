"""Helper utilities for biased_moe_topk_routing."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "num_experts",
    "topk",
    "dtype",
    "index_dtype",
    "renormalize",
    "scoring_func",
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

    for key in ("num_tokens", "num_experts", "topk"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["topk"] > shape["num_experts"]:
        raise ValueError("topk must be less than or equal to num_experts.")
    if not isinstance(shape["renormalize"], bool):
        raise ValueError("renormalize must be a boolean.")
    if shape["scoring_func"] not in ("softmax", "sigmoid"):
        raise ValueError("scoring_func must be 'softmax' or 'sigmoid'.")

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
