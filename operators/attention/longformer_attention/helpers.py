from typing import Any, Dict, Tuple

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "num_heads",
    "head_dim",
    "window_size",
    "dtype",
)

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

    for key in ("batch_size", "seq_len", "num_heads", "head_dim", "window_size"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["head_dim"] % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")

    if shape["window_size"] > shape["seq_len"]:
        raise ValueError("window_size must be less than or equal to seq_len.")

    _resolve_global_indices(shape)
    _resolve_dtype(shape["dtype"])


def _resolve_global_indices(shape: Dict[str, Any]) -> Tuple[int, ...]:
    seq_len = int(shape["seq_len"])
    global_indices = shape.get("global_indices", [0, seq_len - 1])
    if (
        not isinstance(global_indices, (list, tuple))
        or not global_indices
        or not all(isinstance(index, int) for index in global_indices)
    ):
        raise ValueError("global_indices must be a non-empty list or tuple of integers.")
    resolved = tuple(dict.fromkeys(int(index) for index in global_indices))
    if any(index < 0 or index >= seq_len for index in resolved):
        raise ValueError("global_indices must be valid positions within seq_len.")
    return resolved


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
