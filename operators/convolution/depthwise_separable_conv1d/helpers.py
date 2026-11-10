from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "input_dim",
    "depthwise_multiplier",
    "kernel_size",
    "out_channels",
    "padding",
    "dtype",
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

    for key in (
        "batch_size",
        "seq_len",
        "input_dim",
        "depthwise_multiplier",
        "kernel_size",
        "out_channels",
    ):
        _require_positive_int(shape[key], key)
    padding = _require_nonnegative_int(shape["padding"], "padding")
    kernel_size = int(shape["kernel_size"])
    output_seq_len = int(shape["seq_len"]) + 2 * padding - kernel_size + 1
    if output_seq_len <= 0:
        raise ValueError("depthwise_separable_conv1d output sequence length must be positive.")

    _resolve_dtype(shape["dtype"])


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _require_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
