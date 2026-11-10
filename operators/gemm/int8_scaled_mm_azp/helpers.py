"""Helper utilities for int8_scaled_mm_azp."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "M",
    "N",
    "K",
    "has_azp",
    "has_bias",
    "output_dtype",
)

OUTPUT_DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    for key in ("M", "N", "K"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    for key in ("has_azp", "has_bias"):
        if not isinstance(shape[key], bool):
            raise ValueError(f"{key} must be a boolean.")

    _resolve_output_dtype(shape["output_dtype"])


def _resolve_output_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in OUTPUT_DTYPE_NAME_MAP:
        supported = ", ".join(sorted(OUTPUT_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported output_dtype: {dtype_name}. Supported dtypes: {supported}")
    return OUTPUT_DTYPE_NAME_MAP[dtype_name]
