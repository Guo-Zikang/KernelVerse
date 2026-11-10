from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("shape", "num_samples", "dtype")
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
    if len(tensor_shape) > 2:
        raise ValueError("torch.multinomial input must be 1D or 2D.")
    if not isinstance(shape["num_samples"], int) or shape["num_samples"] <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if "replacement" in shape and not isinstance(shape["replacement"], bool):
        raise ValueError("replacement must be a boolean when provided.")
    replacement = bool(shape.get("replacement", False))
    if not replacement and shape["num_samples"] > tensor_shape[-1]:
        raise ValueError("num_samples must be <= shape[-1] when replacement is False.")
    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
