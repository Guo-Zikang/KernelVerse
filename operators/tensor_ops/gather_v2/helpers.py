"""Helper utilities for gather_v2."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "params_shape",
    "indices_shape",
    "axis",
    "dtype",
    "indices_dtype",
)

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    _validate_dims(shape["params_shape"], "params_shape")
    _validate_dims(shape["indices_shape"], "indices_shape")

    axis = int(shape["axis"])
    rank = len(shape["params_shape"])
    if axis < -rank or axis >= rank:
        raise ValueError("axis must be within params rank.")

    _resolve_float_dtype(shape["dtype"])
    _resolve_index_dtype(shape["indices_dtype"])


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _resolve_float_dtype(dtype_name: str) -> torch.dtype:
    dtype = _resolve_dtype(dtype_name)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be FP32, FP16, or BF16.")
    return dtype


def _resolve_index_dtype(dtype_name: str) -> torch.dtype:
    dtype = _resolve_dtype(dtype_name)
    if dtype not in (torch.int32, torch.int64):
        raise ValueError("indices_dtype must be INT32 or INT64.")
    return dtype


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]


def _validate_dims(dims: Any, field_name: str) -> None:
    if not isinstance(dims, (list, tuple)) or not dims:
        raise ValueError(f"{field_name} must be a non-empty list or tuple.")
    for dim in dims:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{field_name} must contain positive integers.")
