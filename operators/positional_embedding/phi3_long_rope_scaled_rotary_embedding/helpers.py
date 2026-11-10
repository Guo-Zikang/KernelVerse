"""Helper utilities for phi3_long_rope_scaled_rotary_embedding."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "num_q_heads",
    "num_kv_heads",
    "head_size",
    "rotary_dim",
    "dtype",
    "original_max_position_embeddings",
    "short_factor",
    "long_factor",
)

DEFAULT_BASE = 10000.0

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
        "batch_size",
        "seq_len",
        "num_q_heads",
        "num_kv_heads",
        "head_size",
        "rotary_dim",
        "original_max_position_embeddings",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["rotary_dim"] > shape["head_size"]:
        raise ValueError("rotary_dim must be less than or equal to head_size.")
    if shape["rotary_dim"] % 2 != 0:
        raise ValueError("rotary_dim must be even.")

    factor_length = shape["rotary_dim"] // 2
    _validate_factor_list("short_factor", shape["short_factor"], factor_length)
    _validate_factor_list("long_factor", shape["long_factor"], factor_length)

    if "base" in shape and float(shape["base"]) <= 0:
        raise ValueError("base must be positive when provided.")
    if "use_long_rope" in shape and not isinstance(shape["use_long_rope"], bool):
        raise ValueError("use_long_rope must be a boolean when provided.")
    if "short_mscale" in shape and float(shape["short_mscale"]) <= 0:
        raise ValueError("short_mscale must be positive when provided.")
    if "long_mscale" in shape and float(shape["long_mscale"]) <= 0:
        raise ValueError("long_mscale must be positive when provided.")

    _resolve_dtype(shape["dtype"])


def _validate_factor_list(name: str, value: Any, expected_length: int) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    if len(value) != expected_length:
        raise ValueError(f"{name} length must equal rotary_dim // 2.")
    for item in value:
        if not isinstance(item, (float, int)) or float(item) <= 0:
            raise ValueError(f"{name} values must be positive numbers.")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
