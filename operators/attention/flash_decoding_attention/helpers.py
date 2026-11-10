from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "query_seq_len",
    "kv_seq_len",
    "num_heads",
    "head_dim",
    "chunk_size",
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
        "kv_seq_len",
        "num_heads",
        "head_dim",
        "chunk_size",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["query_seq_len"] > shape["kv_seq_len"]:
        raise ValueError("query_seq_len must be less than or equal to kv_seq_len.")

    if shape["head_dim"] % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")

    if "causal" in shape and not isinstance(shape["causal"], bool):
        raise ValueError("causal must be a boolean when provided.")

    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
