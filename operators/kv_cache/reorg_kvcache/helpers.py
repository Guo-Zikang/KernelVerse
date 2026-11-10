from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "world_size",
    "num_sequences",
    "feature_dim",
    "padded_chunk_len",
    "chunk_size",
    "chunk_idx",
    "dtype",
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

    for key in ("world_size", "num_sequences", "feature_dim", "padded_chunk_len", "chunk_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if int(shape["chunk_idx"]) < 0:
        raise ValueError("chunk_idx must be non-negative.")

    _resolve_dtype(shape["dtype"])
