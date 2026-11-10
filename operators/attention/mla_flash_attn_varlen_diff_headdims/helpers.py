from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "q_seq_lens",
    "kv_seq_lens",
    "num_heads",
    "qk_head_dim",
    "v_head_dim",
    "dtype",
    "causal",
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

    q_seq_lens = shape["q_seq_lens"]
    kv_seq_lens = shape["kv_seq_lens"]
    if not isinstance(q_seq_lens, (list, tuple)) or not q_seq_lens:
        raise ValueError("q_seq_lens must be a non-empty list or tuple.")
    if not isinstance(kv_seq_lens, (list, tuple)) or not kv_seq_lens:
        raise ValueError("kv_seq_lens must be a non-empty list or tuple.")
    if len(q_seq_lens) != len(kv_seq_lens):
        raise ValueError("q_seq_lens and kv_seq_lens must have the same length.")
    if any(not isinstance(seq_len, int) or seq_len <= 0 for seq_len in q_seq_lens):
        raise ValueError("q_seq_lens must contain positive integers.")
    if any(not isinstance(seq_len, int) or seq_len <= 0 for seq_len in kv_seq_lens):
        raise ValueError("kv_seq_lens must contain positive integers.")

    for key in ("num_heads", "qk_head_dim", "v_head_dim"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    for key in ("qk_head_dim", "v_head_dim"):
        value = int(shape[key])
        if value % 8 != 0:
            raise ValueError(f"{key} must be a multiple of 8.")

    if not isinstance(shape["causal"], bool):
        raise ValueError("causal must be a boolean.")
    if shape["causal"] and any(
        q_seq_len > kv_seq_len
        for q_seq_len, kv_seq_len in zip(q_seq_lens, kv_seq_lens)
    ):
        raise ValueError(
            "For causal attention, every q_seq_len must be less than or equal "
            "to the corresponding kv_seq_len."
        )

    _resolve_dtype(shape["dtype"])
