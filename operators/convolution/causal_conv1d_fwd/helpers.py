from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch",
    "dim",
    "seqlen",
    "width",
    "dtype",
    "use_bias",
    "silu_activation",
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

    for key in ("batch", "dim", "seqlen", "width"):
        _require_positive_int(shape[key], key)
    width = int(shape["width"])
    if width < 2 or width > 4:
        raise ValueError("width must be between 2 and 4.")

    if not isinstance(shape["use_bias"], bool):
        raise ValueError("use_bias must be a boolean.")
    if not isinstance(shape["silu_activation"], bool):
        raise ValueError("silu_activation must be a boolean.")

    _resolve_dtype(shape["dtype"])


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
