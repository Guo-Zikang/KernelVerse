"""Helper utilities for quantize_per_tensor."""

from typing import Any, Dict, Tuple

import torch


REQUIRED_SHAPE_KEYS = ("shape", "scale", "zero_point", "quantized_dtype")

QUANTIZED_DTYPE_NAME_MAP = {
    "QINT8": torch.qint8,
    "QUINT8": torch.quint8,
}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    _validate_dim_sequence(shape["shape"], "shape")

    if not isinstance(shape["scale"], (float, int)) or shape["scale"] <= 0:
        raise ValueError("scale must be a positive number.")
    if not isinstance(shape["zero_point"], int):
        raise ValueError("zero_point must be an integer.")

    quantized_dtype = _resolve_quantized_dtype(shape["quantized_dtype"])
    zero_point = int(shape["zero_point"])
    if quantized_dtype is torch.qint8 and not -128 <= zero_point <= 127:
        raise ValueError("zero_point must be in [-128, 127] for QINT8.")
    if quantized_dtype is torch.quint8 and not 0 <= zero_point <= 255:
        raise ValueError("zero_point must be in [0, 255] for QUINT8.")


def _validate_dim_sequence(value: Any, name: str) -> Tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(dim, int) and dim > 0 for dim in value)
    ):
        raise ValueError(f"{name} must be a non-empty list or tuple of positive integers.")
    return tuple(int(dim) for dim in value)


def _resolve_quantized_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in QUANTIZED_DTYPE_NAME_MAP:
        supported = ", ".join(sorted(QUANTIZED_DTYPE_NAME_MAP.keys()))
        raise ValueError(
            f"Unsupported quantized_dtype: {dtype_name}. Supported dtypes: {supported}"
        )
    return QUANTIZED_DTYPE_NAME_MAP[dtype_name]
