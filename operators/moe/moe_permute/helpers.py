from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "hidden_dim",
    "topk",
    "num_experts",
    "dtype",
    "index_dtype",
)

FLOAT_DTYPE_NAME_MAP = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}

INDEX_DTYPE_NAME_MAP = {
    "INT32": torch.int32,
    "INT64": torch.int64,
}


def _resolve_float_dtype(dtype_name: str) -> torch.dtype:
    try:
        return FLOAT_DTYPE_NAME_MAP[str(dtype_name).upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(FLOAT_DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported floating dtype: {dtype_name}. Supported dtypes: {supported}") from exc


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

    for key in ("num_tokens", "hidden_dim", "topk", "num_experts"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    _resolve_float_dtype(shape["dtype"])
    _resolve_index_dtype(shape["index_dtype"])
