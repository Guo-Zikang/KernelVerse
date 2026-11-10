from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "topk",
    "num_experts",
    "block_size",
    "input_index_dtype",
    "output_index_dtype",
    "pad_sorted_token_ids",
)

INDEX_DTYPE_NAME_MAP = {
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def _resolve_index_dtype(dtype_name: str) -> torch.dtype:
    try:
        return INDEX_DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(INDEX_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported index dtype: {dtype_name}. Supported dtypes: {supported}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    for key in ("num_tokens", "topk", "num_experts", "block_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if not isinstance(shape["pad_sorted_token_ids"], bool):
        raise ValueError("pad_sorted_token_ids must be a boolean.")

    block_size = int(shape["block_size"])
    if block_size & (block_size - 1) != 0:
        raise ValueError("block_size must be a power of 2.")

    _resolve_index_dtype(shape["input_index_dtype"])
    _resolve_index_dtype(shape["output_index_dtype"])


def round_up(value: int, block_size: int) -> int:
    return (int(value) + int(block_size) - 1) // int(block_size) * int(block_size)
