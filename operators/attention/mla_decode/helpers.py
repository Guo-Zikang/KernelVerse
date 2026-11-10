from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "batch_size",
    "seq_len",
    "num_q_heads",
    "block_size",
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
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def validate_shape(shape: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"Missing shape parameter(s): {', '.join(missing)}")

    for key in ("batch_size", "seq_len", "num_q_heads", "block_size"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    block_size = int(shape["block_size"])
    if 128 % block_size != 0:
        raise ValueError("block_size must divide 128 to match MLA block packing.")

    num_q_heads = int(shape["num_q_heads"])
    if num_q_heads > 128:
        raise ValueError("num_q_heads must be in the range [1, 128].")

    _resolve_dtype(shape["dtype"])
