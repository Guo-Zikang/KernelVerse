"""Helper utilities for qk_norm_rope."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "num_heads_q",
    "num_heads_k",
    "num_heads_v",
    "head_dim",
    "rotary_dim",
    "eps",
    "dtype",
    "is_neox",
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

    for key in (
        "num_tokens",
        "num_heads_q",
        "num_heads_k",
        "num_heads_v",
        "head_dim",
        "rotary_dim",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    head_dim = int(shape["head_dim"])
    rotary_dim = int(shape["rotary_dim"])
    if rotary_dim > head_dim:
        raise ValueError("rotary_dim must be less than or equal to head_dim.")
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even.")

    if not isinstance(shape["eps"], (float, int)) or shape["eps"] <= 0:
        raise ValueError("eps must be a positive number.")
    if not isinstance(shape["is_neox"], bool):
        raise ValueError("is_neox must be a boolean.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
