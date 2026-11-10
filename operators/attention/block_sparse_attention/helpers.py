from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "num_heads",
    "head_dim",
    "block_size",
    "num_selected_blocks",
    "selection_mode",
    "dtype",
)
SELECTION_MODES = ("topk", "fixed", "random", "local_plus_topk")
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
        "num_heads",
        "head_dim",
        "block_size",
        "num_selected_blocks",
    ):
        if not isinstance(shape[key], int) or shape[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if shape["head_dim"] % 8 != 0:
        raise ValueError("head_dim must be a positive multiple of 8.")
    if shape["seq_len"] % shape["block_size"] != 0:
        raise ValueError("seq_len must be divisible by block_size.")
    num_blocks = shape["seq_len"] // shape["block_size"]
    if shape["num_selected_blocks"] > num_blocks:
        raise ValueError("num_selected_blocks must be less than or equal to num_blocks.")
    if shape["selection_mode"] not in SELECTION_MODES:
        supported = ", ".join(SELECTION_MODES)
        raise ValueError(f"selection_mode must be one of: {supported}.")
    if "random_seed" in shape and not isinstance(shape["random_seed"], int):
        raise ValueError("random_seed must be an integer when provided.")
    _resolve_dtype(shape["dtype"])


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
