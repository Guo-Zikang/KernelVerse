"""Helper utilities for static_scaled_int8_quant."""

from typing import Any, Dict

import torch

INT8_QMAX = 127.0
MIN_SCALE = 1.0 / (448.0 * 512.0)

REQUIRED_SHAPE_KEYS = (
    "dims",
    "dtype",
)

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    dims = shape["dims"]
    if not isinstance(dims, (list, tuple)) or not dims:
        raise ValueError("dims must be a non-empty list of positive integers.")
    if any((not isinstance(dim, int) or dim <= 0 for dim in dims)):
        raise ValueError("dims must contain positive integers.")

    _resolve_dtype(shape["dtype"])
