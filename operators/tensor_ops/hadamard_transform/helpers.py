"""Helper utilities for hadamard_transform."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("shape", "normalize", "dtype")
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

    tensor_shape = shape["shape"]
    if (
        not isinstance(tensor_shape, (list, tuple))
        or not tensor_shape
        or not all(isinstance(dim, int) and dim > 0 for dim in tensor_shape)
    ):
        raise ValueError("shape must be a non-empty list or tuple of positive integers.")
    if not _is_power_of_two(int(tensor_shape[-1])):
        raise ValueError("The last dimension must be a power of two.")
    if not isinstance(shape["normalize"], bool):
        raise ValueError("normalize must be a boolean.")

    _resolve_dtype(shape["dtype"])


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
