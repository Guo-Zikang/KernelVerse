from typing import Any, Dict, Tuple

import torch

REQUIRED_SHAPE_KEYS = ("shape", "perm", "dtype")
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

    tensor_shape = shape["shape"]
    perm = shape["perm"]
    if (
        not isinstance(tensor_shape, (list, tuple))
        or not tensor_shape
        or not all(isinstance(dim, int) and dim > 0 for dim in tensor_shape)
    ):
        raise ValueError("shape must be a non-empty list or tuple of positive integers.")
    if not isinstance(perm, (list, tuple)) or len(perm) != len(tensor_shape):
        raise ValueError("perm must be a list or tuple with the same length as shape.")
    if sorted(perm) != list(range(len(tensor_shape))):
        raise ValueError("perm must be a valid permutation of dimension indices.")

    _resolve_dtype(shape["dtype"])


def _normalize_perm(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in perm)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
