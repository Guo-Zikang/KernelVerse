from typing import Any, Dict, Tuple, Union

import torch


REQUIRED_SHAPE_KEYS = ("shape", "dim", "dtype")
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
    tensor_shape = shape["shape"]
    if (
        not isinstance(tensor_shape, (list, tuple))
        or not tensor_shape
        or not all(isinstance(dim, int) and dim > 0 for dim in tensor_shape)
    ):
        raise ValueError("shape must be a non-empty list or tuple of positive integers.")
    rank = len(tensor_shape)
    _validate_dim(shape["dim"], rank)
    if "keepdim" in shape and not isinstance(shape["keepdim"], bool):
        raise ValueError("keepdim must be a boolean when provided.")
    _resolve_dtype(shape["dtype"])


def _normalize_dim(dim: Union[int, Tuple[int, ...], list]) -> Union[int, Tuple[int, ...]]:
    if isinstance(dim, int):
        return int(dim)
    if isinstance(dim, (list, tuple)) and dim:
        return tuple(int(value) for value in dim)
    raise ValueError("dim must be an integer or a non-empty list/tuple of integers.")


def _validate_dim(dim: Any, rank: int) -> None:
    normalized_dim = _normalize_dim(dim)
    dims = (normalized_dim,) if isinstance(normalized_dim, int) else normalized_dim
    normalized_positive_dims = []
    for value in dims:
        if not isinstance(value, int):
            raise ValueError("dim entries must be integers.")
        if value < -rank or value >= rank:
            raise ValueError(f"dim entries must be in range [{-rank}, {rank - 1}].")
        normalized_positive_dims.append(value + rank if value < 0 else value)
    if len(set(normalized_positive_dims)) != len(normalized_positive_dims):
        raise ValueError("dim entries must refer to unique tensor dimensions.")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_NAME_MAP:
        supported = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(f"Unsupported dtype: {dtype_name}. Supported dtypes: {supported}")
    return DTYPE_NAME_MAP[dtype_name]
