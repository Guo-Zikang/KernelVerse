"""Helper utilities for relative_position_bias_attention."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("batch_size", "seq_len", "num_heads", "head_dim", "dtype")

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

    for key in ("batch_size", "seq_len", "num_heads", "head_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["head_dim"] % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")
    if "num_buckets" in shape and (
        not isinstance(shape["num_buckets"], int) or shape["num_buckets"] <= 1
    ):
        raise ValueError("num_buckets must be an integer greater than 1 when provided.")
    if "max_distance" in shape and (
        not isinstance(shape["max_distance"], int) or shape["max_distance"] <= 0
    ):
        raise ValueError("max_distance must be a positive integer when provided.")
    if "bias" in shape and not isinstance(shape["bias"], bool):
        raise ValueError("bias must be a boolean when provided.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
