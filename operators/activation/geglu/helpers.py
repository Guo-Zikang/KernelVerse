from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = ("shape", "dim", "dtype")
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
    if not isinstance(shape["dim"], int):
        raise ValueError("dim must be an integer.")
    dim = int(shape["dim"])
    rank = len(tensor_shape)
    if dim < -rank or dim >= rank:
        raise ValueError(f"dim must be in range [{-rank}, {rank - 1}].")
    normalized_dim = dim + rank if dim < 0 else dim
    if tensor_shape[normalized_dim] % 2 != 0:
        raise ValueError("shape[dim] must be even for GEGLU.")
    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
