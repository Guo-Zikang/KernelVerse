from typing import Any, Dict

import torch


REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "in_channels",
    "out_channels",
    "height",
    "width",
    "kernel_size",
    "stride",
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

    for key in ("batch_size", "in_channels", "out_channels", "height", "width"):
        _require_positive_int(shape[key], key)
    kernel_size = _require_positive_int(shape["kernel_size"], "kernel_size")
    stride = _require_positive_int(shape["stride"], "stride")

    output_height = (int(shape["height"]) - kernel_size) // stride + 1
    padded_width = int(shape["width"]) + kernel_size - 1 + stride - 1
    output_width = (padded_width - kernel_size) // stride + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("causal_conv2d output height and width must be positive.")

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
