"""Random shape generation for benchmark shape templates."""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Dict, List

from operators.registry import load_module


RANDOM_SHAPE_SPACE_NAME = "RANDOM_SHAPE_SPACE"


def generate_random_seed() -> int:
    """Generate a reproducible seed value when the user does not provide one."""
    return time.time_ns() % (2**31)


def _copy_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a JSON-like shape dictionary."""
    return json.loads(json.dumps(shape))


def _shape_key(shape: Dict[str, Any]) -> str:
    """Build a stable key used for de-duplicating generated shapes."""
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


def _load_random_shape_space(shape_module_name: str) -> Dict[str, List[Any]]:
    """Load random shape candidate space from a shape module."""
    shape_module = load_module(shape_module_name)
    random_shape_space = getattr(shape_module, RANDOM_SHAPE_SPACE_NAME, None)

    if random_shape_space is None:
        raise ValueError(
            f"Shape template module {shape_module_name} must define "
            f"{RANDOM_SHAPE_SPACE_NAME} for shape_mode=random."
        )
    if not isinstance(random_shape_space, dict) or not random_shape_space:
        raise ValueError(f"{shape_module_name}.{RANDOM_SHAPE_SPACE_NAME} must be a non-empty dict.")

    for parameter_name, candidates in random_shape_space.items():
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError(
                f"{shape_module_name}.{RANDOM_SHAPE_SPACE_NAME}[{parameter_name!r}] "
                "must be a non-empty list or tuple."
            )

    return random_shape_space


def resolve_random_shapes(
    shape_module_name: str,
    *,
    seed: int,
    count: int,
    max_attempts: int,
    validate_shape_fn: Callable[[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    """Generate unique random shapes from a shape module's RANDOM_SHAPE_SPACE."""
    if count <= 0:
        raise ValueError("random_count must be a positive integer.")
    if max_attempts < count:
        raise ValueError("random_max_attempts must be greater than or equal to random_count.")

    random_shape_space = _load_random_shape_space(shape_module_name)
    rng = random.Random(seed)
    generated_shapes: List[Dict[str, Any]] = []
    seen_shapes = set()

    for _ in range(max_attempts):
        candidate = {
            parameter_name: _copy_shape(rng.choice(list(candidates)))
            for parameter_name, candidates in random_shape_space.items()
        }
        try:
            validate_shape_fn(candidate)
        except Exception:
            continue

        candidate_key = _shape_key(candidate)
        if candidate_key in seen_shapes:
            continue
        seen_shapes.add(candidate_key)
        generated_shapes.append(candidate)
        if len(generated_shapes) == count:
            return generated_shapes

    raise ValueError(
        f"Could only generate {len(generated_shapes)} unique random shape(s) from "
        f"{shape_module_name} after {max_attempts} attempts; requested {count}."
    )
