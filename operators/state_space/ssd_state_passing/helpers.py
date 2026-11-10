"""Helper utilities for ssd_state_passing."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "num_chunks",
    "num_heads",
    "state_dim",
    "chunk_size",
    "dtype",
    "has_initial_states",
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

    for key in ("batch_size", "num_chunks", "num_heads", "state_dim", "chunk_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if int(shape["batch_size"]) > int(shape["num_chunks"]):
        raise ValueError("batch_size must be less than or equal to num_chunks.")

    if not isinstance(shape["has_initial_states"], bool):
        raise ValueError("has_initial_states must be a boolean.")

    _resolve_dtype(shape["dtype"])
