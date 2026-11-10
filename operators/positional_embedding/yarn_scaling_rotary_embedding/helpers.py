"""Helper utilities for yarn_scaling_rotary_embedding."""

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
    "scaling_factor",
    "max_position_embeddings",
)

DEFAULT_BASE = 10000.0
DEFAULT_IS_NEOX_STYLE = True

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
        "max_position_embeddings",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["rotary_dim"] > shape["head_size"]:
        raise ValueError("rotary_dim must be less than or equal to head_size.")
    if shape["rotary_dim"] % 2 != 0:
        raise ValueError("rotary_dim must be even.")
    if float(shape["scaling_factor"]) <= 0:
        raise ValueError("scaling_factor must be positive.")

    for key in ("base", "extrapolation_factor", "attn_factor"):
        if key in shape and not isinstance(shape[key], (float, int)):
            raise ValueError(f"{key} must be a number when provided.")
    for key in ("beta_fast", "beta_slow"):
        if key in shape and (not isinstance(shape[key], int) or shape[key] <= 0):
            raise ValueError(f"{key} must be a positive integer when provided.")
    if "truncate" in shape and not isinstance(shape["truncate"], bool):
        raise ValueError("truncate must be a boolean when provided.")
    if "is_neox_style" in shape and not isinstance(shape["is_neox_style"], bool):
        raise ValueError("is_neox_style must be a boolean when provided.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
