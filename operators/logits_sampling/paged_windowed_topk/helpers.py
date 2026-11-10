"""Helper utilities for paged_windowed_topk."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch",
    "length",
    "window_length",
    "topk",
    "dtype",
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

    for key in ("batch", "length", "window_length", "topk"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["window_length"] > shape["length"]:
        raise ValueError("window_length must be less than or equal to length.")
    if shape["topk"] > shape["window_length"]:
        raise ValueError("topk must be less than or equal to window_length.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
