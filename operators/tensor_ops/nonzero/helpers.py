from typing import Any, Dict, Tuple

import torch

REQUIRED_SHAPE_KEYS = (
    "input_dims",
    "dtype",
    "index_dtype",
    "target_nonzero_count",
)

INPUT_DTYPE_NAME_MAP = {
    "BOOL": torch.bool,
    "INT32": torch.int32,
    "INT64": torch.int64,
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}

INDEX_DTYPE_NAME_MAP = {
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def _resolve_input_dtype(dtype_name: str) -> torch.dtype:
    try:
        return INPUT_DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(INPUT_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported input dtype: {dtype_name}. Supported dtypes: {supported}") from exc


def _resolve_index_dtype(dtype_name: str) -> torch.dtype:
    try:
        return INDEX_DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(INDEX_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported index dtype: {dtype_name}. Supported dtypes: {supported}") from exc


def _normalize_dims(dims: Any) -> Tuple[int, ...]:
    if (
        not isinstance(dims, (list, tuple))
        or not dims
        or not all(isinstance(dim, int) and dim > 0 for dim in dims)
    ):
        raise ValueError("input_dims must be a non-empty list or tuple of positive integers.")
    return tuple(int(dim) for dim in dims)


def _numel(dims: Tuple[int, ...]) -> int:
    total = 1
    for dim in dims:
        total *= dim
    return total


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    input_dims = _normalize_dims(shape["input_dims"])
    target_nonzero_count = int(shape["target_nonzero_count"])
    if target_nonzero_count < 0:
        raise ValueError("target_nonzero_count must be non-negative.")
    if target_nonzero_count > _numel(input_dims):
        raise ValueError("target_nonzero_count cannot exceed the number of input elements.")

    _resolve_input_dtype(shape["dtype"])
    _resolve_index_dtype(shape["index_dtype"])
