"""Helper utilities for concat_and_cache_mla_rope."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "num_blocks",
    "block_size",
    "latent_dim",
    "rope_dim",
    "max_position",
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

    for key in ("num_tokens", "num_blocks", "block_size", "latent_dim", "rope_dim", "max_position"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if int(shape["rope_dim"]) % 2 != 0:
        raise ValueError("rope_dim must be even.")

    if int(shape["num_tokens"]) > int(shape["num_blocks"]) * int(shape["block_size"]):
        raise ValueError("num_tokens must not exceed num_blocks * block_size.")

    if "is_neox" in shape and not isinstance(shape["is_neox"], bool):
        raise ValueError("is_neox must be a boolean when provided.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
