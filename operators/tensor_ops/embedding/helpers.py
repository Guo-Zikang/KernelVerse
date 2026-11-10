"""Helper utilities for embedding."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("input_shape", "num_embeddings", "embedding_dim", "dtype")

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

    input_shape = shape["input_shape"]
    if (
        not isinstance(input_shape, (list, tuple))
        or not input_shape
        or not all(isinstance(dim, int) and dim > 0 for dim in input_shape)
    ):
        raise ValueError(
            "input_shape must be a non-empty list or tuple of positive integers."
        )

    for key in ("num_embeddings", "embedding_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
