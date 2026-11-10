"""Helper utilities for flash_mla_sparse."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("seq_q", "seq_kv", "num_heads", "head_dim", "topk", "dtype")
DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    for key in ("seq_q", "seq_kv", "num_heads", "head_dim", "topk"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if int(shape["head_dim"]) % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")
    if int(shape["topk"]) > int(shape["seq_kv"]):
        raise ValueError("topk must be less than or equal to seq_kv.")

    _resolve_dtype(shape["dtype"])
    if "softmax_scale" in shape and not isinstance(shape["softmax_scale"], (int, float)):
        raise ValueError("softmax_scale must be a number when provided.")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
