"""Helper utilities for ssd_bmm."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "seq_lens",
    "num_groups",
    "key_dim",
    "chunk_size",
    "dtype",
    "causal",
)

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    seq_lens = shape["seq_lens"]
    if not isinstance(seq_lens, (list, tuple)) or not seq_lens:
        raise ValueError("seq_lens must be a non-empty list of positive integers.")
    if any((not isinstance(seq_len, int) or seq_len <= 0 for seq_len in seq_lens)):
        raise ValueError("seq_lens must contain positive integers.")

    for key in ("num_groups", "key_dim", "chunk_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if not isinstance(shape["causal"], bool):
        raise ValueError("causal must be a boolean.")

    _resolve_dtype(shape["dtype"])
