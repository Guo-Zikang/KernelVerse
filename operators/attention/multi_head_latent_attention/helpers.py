from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "query_seq_len",
    "num_q_heads",
    "qk_head_dim",
    "v_head_dim",
    "num_blocks",
    "block_size",
    "max_blocks_per_seq",
    "cache_seq_len",
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

    for key in (
        "batch_size",
        "query_seq_len",
        "num_q_heads",
        "qk_head_dim",
        "v_head_dim",
        "num_blocks",
        "block_size",
        "max_blocks_per_seq",
        "cache_seq_len",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["v_head_dim"] > shape["qk_head_dim"]:
        raise ValueError("v_head_dim must be less than or equal to qk_head_dim.")

    if shape["query_seq_len"] > shape["cache_seq_len"]:
        raise ValueError("query_seq_len must be less than or equal to cache_seq_len.")

    blocks_needed = (shape["cache_seq_len"] + shape["block_size"] - 1) // shape[
        "block_size"
    ]
    if shape["max_blocks_per_seq"] < blocks_needed:
        raise ValueError("max_blocks_per_seq is too small for cache_seq_len.")

    if shape["num_blocks"] < shape["batch_size"] * blocks_needed:
        raise ValueError("num_blocks must cover every batch sequence block.")

    if "causal" in shape and not isinstance(shape["causal"], bool):
        raise ValueError("causal must be a boolean when provided.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
