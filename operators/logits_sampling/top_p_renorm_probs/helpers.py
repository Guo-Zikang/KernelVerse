"""Helper utilities for top_p_renorm_probs."""

from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "vocab_size",
    "top_p",
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

    for key in ("batch_size", "vocab_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    top_p = float(shape["top_p"])
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")

    _resolve_dtype(shape["dtype"])
