"""Helper utilities for chunk_cumsum_dt."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "seq_lens",
    "num_heads",
    "chunk_size",
    "dtype",
    "dt_softplus",
    "has_dt_bias",
)

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

    if not isinstance(shape["seq_lens"], (list, tuple)) or not shape["seq_lens"]:
        raise ValueError("seq_lens must be a non-empty list or tuple of positive integers.")
    if any((not isinstance(seq_len, int) or seq_len <= 0 for seq_len in shape["seq_lens"])):
        raise ValueError("seq_lens must contain positive integers.")

    for key in ("num_heads", "chunk_size"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if not isinstance(shape["dt_softplus"], bool):
        raise ValueError("dt_softplus must be a boolean.")
    if not isinstance(shape["has_dt_bias"], bool):
        raise ValueError("has_dt_bias must be a boolean.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
