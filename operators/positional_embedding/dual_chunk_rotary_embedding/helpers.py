"""Helper utilities for dual_chunk_rotary_embedding."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "num_q_heads",
    "num_kv_heads",
    "head_size",
    "rotary_dim",
    "dtype",
    "chunk_size",
    "local_size",
)

DEFAULT_BASE = 10000.0
DEFAULT_IS_NEOX_STYLE = True

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

    for key in (
        "batch_size",
        "seq_len",
        "num_q_heads",
        "num_kv_heads",
        "head_size",
        "rotary_dim",
        "chunk_size",
        "local_size",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["rotary_dim"] > shape["head_size"]:
        raise ValueError("rotary_dim must be less than or equal to head_size.")
    if shape["rotary_dim"] % 2 != 0:
        raise ValueError("rotary_dim must be even.")
    if shape["local_size"] >= shape["chunk_size"]:
        raise ValueError("local_size must be smaller than chunk_size.")
    if "base" in shape and not isinstance(shape["base"], (float, int)):
        raise ValueError("base must be a number when provided.")
    if "is_neox_style" in shape and not isinstance(shape["is_neox_style"], bool):
        raise ValueError("is_neox_style must be a boolean when provided.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
