from typing import Any, Dict

import torch

REQUIRED_SHAPE_KEYS = (
    "num_tokens",
    "num_experts",
    "num_expert_group",
    "topk_group",
    "topk",
    "num_fused_shared_experts",
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

    for key in ("num_tokens", "num_experts", "num_expert_group", "topk_group", "topk"):
        value = int(shape[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    num_fused_shared_experts = int(shape["num_fused_shared_experts"])
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative.")

    num_experts = int(shape["num_experts"])
    num_expert_group = int(shape["num_expert_group"])
    topk_group = int(shape["topk_group"])
    topk = int(shape["topk"])

    if num_experts % num_expert_group != 0:
        raise ValueError("num_experts must be divisible by num_expert_group.")
    if topk_group > num_expert_group:
        raise ValueError("topk_group must be less than or equal to num_expert_group.")
    if topk <= num_fused_shared_experts:
        raise ValueError("topk must be greater than num_fused_shared_experts.")
    if topk > num_experts + num_fused_shared_experts:
        raise ValueError("topk is too large for available experts.")
    experts_per_group = num_experts // num_expert_group
    if experts_per_group < 2:
        raise ValueError("Each expert group must contain at least 2 experts.")
    if topk - num_fused_shared_experts > topk_group * experts_per_group:
        raise ValueError("routed topk exceeds the selected expert capacity.")

    _resolve_float_dtype(shape["dtype"])
    _resolve_index_dtype(shape["index_dtype"])
