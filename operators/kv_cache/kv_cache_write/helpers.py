"""Helper utilities for kv_cache_write."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "num_kv_heads",
    "max_seq_len",
    "write_seq_len",
    "head_dim",
    "dtype",
)
DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}
POSITION_MODES = {"shared", "per_batch"}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    for key in ("batch_size", "num_kv_heads", "max_seq_len", "write_seq_len", "head_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if shape["write_seq_len"] > shape["max_seq_len"]:
        raise ValueError("write_seq_len must be less than or equal to max_seq_len.")

    position_mode = shape.get("position_mode", "shared")
    if position_mode not in POSITION_MODES:
        supported = ", ".join(sorted(POSITION_MODES))
        raise ValueError(f"position_mode must be one of: {supported}.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
