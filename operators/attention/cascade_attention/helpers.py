from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "shared_len",
    "unique_lens",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
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

    for key in ("batch_size", "shared_len", "num_q_heads", "num_kv_heads", "head_dim"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    unique_lens = shape["unique_lens"]
    if (
        not isinstance(unique_lens, (list, tuple))
        or len(unique_lens) != int(shape["batch_size"])
        or not all(isinstance(seq_len, int) and seq_len >= 0 for seq_len in unique_lens)
    ):
        raise ValueError("unique_lens must be a list of non-negative integers with length batch_size.")

    if int(shape["num_q_heads"]) % int(shape["num_kv_heads"]) != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads.")

    if int(shape["head_dim"]) % 8 != 0:
        raise ValueError("head_dim must be a multiple of 8.")

    if "softmax_scale" in shape and not isinstance(shape["softmax_scale"], (int, float)):
        raise ValueError("softmax_scale must be numeric when provided.")

    _resolve_dtype(shape["dtype"])
