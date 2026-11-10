from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "seq_lens",
    "num_heads",
    "num_groups",
    "head_dim",
    "dstate",
    "chunk_size",
    "dtype",
    "dt_softplus",
    "has_dt_bias",
    "has_D",
    "has_z",
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
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    seq_lens = shape["seq_lens"]
    if not isinstance(seq_lens, (list, tuple)) or not seq_lens:
        raise ValueError("seq_lens must be a non-empty list of positive integers.")
    if any(not isinstance(seq_len, int) or seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must contain positive integers.")

    for key in ("num_heads", "num_groups", "head_dim", "dstate", "chunk_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if int(shape["num_heads"]) % int(shape["num_groups"]) != 0:
        raise ValueError("num_heads must be divisible by num_groups.")

    if int(shape["head_dim"]) % 8 != 0:
        raise ValueError("head_dim must be a multiple of 8.")

    if int(shape["dstate"]) <= 0:
        raise ValueError("dstate must be positive.")

    for key in ("dt_softplus", "has_dt_bias", "has_D", "has_z", "has_initial_states"):
        if not isinstance(shape[key], bool):
            raise ValueError(f"{key} must be a boolean.")

    _resolve_dtype(shape["dtype"])
