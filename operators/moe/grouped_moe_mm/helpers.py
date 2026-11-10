"""Helper utilities for grouped_moe_mm."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_experts",
    "tokens_per_expert",
    "K",
    "N",
    "per_act_token",
    "per_out_ch",
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

    for key in ("num_experts", "tokens_per_expert", "K", "N"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    for key in ("per_act_token", "per_out_ch"):
        if not isinstance(shape[key], bool):
            raise ValueError(f"{key} must be a boolean.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
