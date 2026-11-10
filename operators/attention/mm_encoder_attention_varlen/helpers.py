from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = ("seq_lens", "num_heads", "head_dim", "dtype")

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

    seq_lens = shape["seq_lens"]
    if not isinstance(seq_lens, (list, tuple)) or not seq_lens:
        raise ValueError("seq_lens must be a non-empty list or tuple.")
    if any(not isinstance(seq_len, int) or seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must contain positive integers.")

    for key in ("num_heads", "head_dim"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if int(shape["head_dim"]) % 8 != 0:
        raise ValueError("head_dim must be a multiple of 8.")

    _resolve_dtype(shape["dtype"])
