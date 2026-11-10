from typing import Any, Dict, Sequence, Tuple

import torch

REQUIRED_SHAPE_KEYS = ("lhs_dims", "rhs_dims", "dtype")

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}") from exc


def _normalize_dims(dims: Any, field_name: str) -> Tuple[int, ...]:
    if (
        not isinstance(dims, (list, tuple))
        or not dims
        or not all(isinstance(dim, int) and dim > 0 for dim in dims)
    ):
        raise ValueError(f"{field_name} must be a non-empty list or tuple of positive integers.")
    return tuple(int(dim) for dim in dims)


def _broadcast_shape(lhs_dims: Sequence[int], rhs_dims: Sequence[int]) -> Tuple[int, ...]:
    lhs = list(lhs_dims)
    rhs = list(rhs_dims)
    rank = max(len(lhs), len(rhs))
    lhs = [1] * (rank - len(lhs)) + lhs
    rhs = [1] * (rank - len(rhs)) + rhs
    output = []
    for lhs_dim, rhs_dim in zip(lhs, rhs):
        if lhs_dim == rhs_dim or lhs_dim == 1 or rhs_dim == 1:
            output.append(max(lhs_dim, rhs_dim))
        else:
            raise ValueError("lhs_dims and rhs_dims are not broadcast-compatible.")
    return tuple(output)


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    lhs_dims = _normalize_dims(shape["lhs_dims"], "lhs_dims")
    rhs_dims = _normalize_dims(shape["rhs_dims"], "rhs_dims")
    _broadcast_shape(lhs_dims, rhs_dims)
    _resolve_dtype(shape["dtype"])
