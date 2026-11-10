"""Helper utilities for batched_moe_align_block_size."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "max_tokens_per_batch",
    "num_experts",
    "block_size",
    "output_index_dtype",
    "simulate_empty_batches",
)

INDEX_DTYPE_NAME_MAP = {
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def validate_shape(shape: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing shape parameter(s): {missing}")

    for key in ("max_tokens_per_batch", "num_experts", "block_size"):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if not isinstance(shape["simulate_empty_batches"], bool):
        raise ValueError("simulate_empty_batches must be a boolean.")

    _resolve_index_dtype(shape["output_index_dtype"])


def _resolve_index_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in INDEX_DTYPE_NAME_MAP:
        supported = ", ".join(sorted(INDEX_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported index dtype: {dtype_name}. Supported dtypes: {supported}")
    return INDEX_DTYPE_NAME_MAP[dtype_name]


def round_up(value: int, block_size: int) -> int:
    return (int(value) + int(block_size) - 1) // int(block_size) * int(block_size)
