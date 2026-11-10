"""Helper utilities for add_rmsnorm."""

from typing import Any, Dict, Tuple

import torch


REQUIRED_SHAPE_KEYS = ("shape", "eps", "dtype")

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

    _validate_dim_sequence(shape["shape"], "shape")

    if not isinstance(shape["eps"], (float, int)) or shape["eps"] <= 0:
        raise ValueError("eps must be a positive number.")

    _resolve_dtype(shape["dtype"])


def _validate_dim_sequence(value: Any, name: str) -> Tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(dim, int) and dim > 0 for dim in value)
    ):
        raise ValueError(f"{name} must be a non-empty list or tuple of positive integers.")
    return tuple(int(dim) for dim in value)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
