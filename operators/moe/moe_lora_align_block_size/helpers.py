from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "topk",
    "num_experts",
    "block_size",
    "max_loras",
    "max_num_tokens_padded",
    "max_num_m_blocks",
    "input_index_dtype",
    "output_index_dtype",
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


def _round_up(value: int, block_size: int) -> int:
    return (int(value) + int(block_size) - 1) // int(block_size) * int(block_size)


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    for key in (
        "num_tokens",
        "topk",
        "num_experts",
        "block_size",
        "max_loras",
        "max_num_tokens_padded",
        "max_num_m_blocks",
    ):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    block_size = int(shape["block_size"])
    if block_size & (block_size - 1) != 0:
        raise ValueError("block_size must be a power of 2.")

    num_tokens = int(shape["num_tokens"])
    topk = int(shape["topk"])
    max_num_tokens_padded = int(shape["max_num_tokens_padded"])
    max_num_m_blocks = int(shape["max_num_m_blocks"])
    minimum_padded = _round_up(num_tokens * topk, block_size)
    if max_num_tokens_padded < minimum_padded:
        raise ValueError("max_num_tokens_padded is too small for num_tokens * topk.")
    if max_num_tokens_padded % block_size != 0:
        raise ValueError("max_num_tokens_padded must be divisible by block_size.")
    if max_num_m_blocks < max_num_tokens_padded // block_size:
        raise ValueError("max_num_m_blocks is too small for max_num_tokens_padded.")

    _resolve_index_dtype(shape["input_index_dtype"])
    _resolve_index_dtype(shape["output_index_dtype"])
