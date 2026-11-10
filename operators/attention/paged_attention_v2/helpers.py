"""Helper utilities for paged_attention_v2."""

from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "partition_size",
    "input_dtype",
    "output_dtype",
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
        "seq_len",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "block_size",
        "partition_size",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if shape["num_q_heads"] % shape["num_kv_heads"] != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads.")
    if shape["head_dim"] % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")

    if "softmax_scale" in shape and (
        not isinstance(shape["softmax_scale"], (float, int))
        or shape["softmax_scale"] <= 0
    ):
        raise ValueError("softmax_scale must be a positive number when provided.")

    _resolve_dtype(str(shape["input_dtype"]))
    _resolve_dtype(str(shape["output_dtype"]))


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
