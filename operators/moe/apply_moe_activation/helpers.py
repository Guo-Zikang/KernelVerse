"""Helper utilities for apply_moe_activation."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "hidden_dim",
    "activation",
    "dtype",
)

GLU_ACTIVATIONS = {"silu", "gelu", "relu2", "swigluoai", "swiglustep"}
NO_MUL_ACTIVATIONS = {"silu_no_mul", "gelu_no_mul", "relu2_no_mul"}
VALID_ACTIVATIONS = GLU_ACTIVATIONS | NO_MUL_ACTIVATIONS

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

    for key in ("num_tokens", "hidden_dim"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["activation"] not in VALID_ACTIVATIONS:
        supported = ", ".join(sorted(VALID_ACTIVATIONS))
        raise ValueError(f"Unsupported activation: {shape['activation']}. Supported activations: {supported}")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]


def activation_input_width(hidden_dim: int, activation: str) -> int:
    return hidden_dim if activation in NO_MUL_ACTIVATIONS else hidden_dim * 2
