"""Helper utilities for flash_attn_varlen."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("seq_lens", "num_heads", "head_dim", "dtype")

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

    seq_lens = shape["seq_lens"]
    if (
        not isinstance(seq_lens, (list, tuple))
        or not seq_lens
        or not all(isinstance(seq_len, int) and seq_len > 0 for seq_len in seq_lens)
    ):
        raise ValueError("seq_lens must be a non-empty list of positive integers.")

    for key in ("num_heads", "head_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if int(shape["head_dim"]) % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")

    _resolve_dtype(shape["dtype"])
    if "softmax_scale" in shape and not isinstance(shape["softmax_scale"], (int, float)):
        raise ValueError("softmax_scale must be a number when provided.")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
