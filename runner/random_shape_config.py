"""Operator-level random shape configuration helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


DEFAULT_RANDOM_COUNT = 100
DEFAULT_RANDOM_MAX_ATTEMPTS = 10000


def resolve_random_shape_settings(
    operator_spec: Dict[str, Any],
    *,
    random_count: Optional[int] = None,
    random_max_attempts: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve random shape count and max attempts from CLI overrides or operator config."""
    config = operator_spec.get("config") or {}
    random_shape_config = config.get("random_shape") or {}
    if not isinstance(random_shape_config, dict):
        raise ValueError("operator config random_shape must be an object when provided.")

    resolved_count = (
        random_count
        if random_count is not None
        else random_shape_config.get("count", DEFAULT_RANDOM_COUNT)
    )
    resolved_max_attempts = (
        random_max_attempts
        if random_max_attempts is not None
        else random_shape_config.get("max_attempts", DEFAULT_RANDOM_MAX_ATTEMPTS)
    )

    resolved_count = int(resolved_count)
    resolved_max_attempts = int(resolved_max_attempts)
    if resolved_count < 1:
        raise ValueError("random shape count must be >= 1.")
    if resolved_max_attempts < resolved_count:
        raise ValueError("random shape max_attempts must be greater than or equal to count.")
    return resolved_count, resolved_max_attempts
