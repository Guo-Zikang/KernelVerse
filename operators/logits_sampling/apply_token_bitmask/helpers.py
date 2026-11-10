"""Helper utilities for apply_token_bitmask."""

from typing import Any, Dict

import torch

BITS_PER_BLOCK = 32
UINT32_MASK = 0xFFFFFFFF

REQUIRED_SHAPE_KEYS = (
    "batch",
    "vocab_size",
    "dtype",
    "use_indices",
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

    for key in ("batch", "vocab_size"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if int(shape["vocab_size"]) % BITS_PER_BLOCK != 0:
        raise ValueError("vocab_size must be divisible by 32.")

    if not isinstance(shape["use_indices"], bool):
        raise ValueError("use_indices must be a boolean.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
