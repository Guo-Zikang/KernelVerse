from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = ("shape", "dtype")

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    tensor_shape = shape["shape"]
    if not isinstance(tensor_shape, (list, tuple)) or len(tensor_shape) < 1:
        raise ValueError("shape must be a non-empty list or tuple.")
    if any(not isinstance(dim, int) or dim <= 0 for dim in tensor_shape):
        raise ValueError("shape dimensions must be positive integers.")

    _resolve_dtype(shape["dtype"])
