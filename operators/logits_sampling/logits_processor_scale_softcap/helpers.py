from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = ("num_seqs", "vocab_size", "dtype")

DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    for key in ("num_seqs", "vocab_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    scale = float(shape.get("scale", 1.0))
    if scale < 0:
        raise ValueError("scale must be non-negative.")

    soft_cap = shape.get("soft_cap")
    if soft_cap is not None and float(soft_cap) <= 0:
        raise ValueError("soft_cap must be positive or None.")

    _resolve_dtype(shape["dtype"])
