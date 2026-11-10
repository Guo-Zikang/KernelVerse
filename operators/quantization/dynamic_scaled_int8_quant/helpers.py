"""Helper utilities for dynamic_scaled_int8_quant."""

from typing import Any, Dict

import torch

INT8_QMAX = 127.0
MIN_SCALING_FACTOR = 1.0 / (448.0 * 512.0)

REQUIRED_SHAPE_KEYS = (
    "dims",
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

    dims = shape["dims"]
    if not isinstance(dims, (list, tuple)) or len(dims) == 0:
        raise ValueError("dims must be a non-empty shape.")
    if any((not isinstance(dim, int)) or dim <= 0 for dim in dims):
        raise ValueError("dims must contain positive integers.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
