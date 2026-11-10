"""Category-aware correctness policy resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


POLICY_STRICT = "strict"
POLICY_TOLERANT = "tolerant"
POLICY_COSINE = "cosine"


CATEGORY_DEFAULT_POLICIES = {
    "activation": POLICY_STRICT,
    "attention": POLICY_COSINE,
    "convolution": POLICY_TOLERANT,
    "gemm": POLICY_TOLERANT,
    "kv_cache": POLICY_STRICT,
    "logits_sampling": POLICY_TOLERANT,
    "moe": POLICY_TOLERANT,
    "normalization": POLICY_TOLERANT,
    "positional_embedding": POLICY_COSINE,
    "quantization": POLICY_STRICT,
    "reduction": POLICY_TOLERANT,
    "state_space": POLICY_COSINE,
    "tensor_ops": POLICY_STRICT,
}


TOLERANT_THRESHOLDS = {
    "FP32": {
        "min_cosine": 0.9990,
        "max_mismatch_rate": 0.01,
        "max_p99_abs": 1e-4,
        "max_p99_rel": 1e-4,
    },
    "FP16": {
        "min_cosine": 0.9900,
        "max_mismatch_rate": 0.02,
        "max_p99_abs": 0.25,
        "max_p99_rel": 0.03,
        "near_zero_abs": 0.25,
        "max_near_zero_p99_abs": 1.0,
        "max_near_zero_mean_abs": 0.05,
        "max_near_zero_bad_abs": 2.0,
        "max_near_zero_bad_rate": 0.01,
    },
    "BF16": {
        "min_cosine": 0.9900,
        "max_mismatch_rate": 0.03,
        "max_p99_abs": 0.5,
        "max_p99_rel": 0.05,
        "near_zero_abs": 0.25,
        "max_near_zero_p99_abs": 1.0,
        "max_near_zero_mean_abs": 0.05,
        "max_near_zero_bad_abs": 2.0,
        "max_near_zero_bad_rate": 0.01,
    },
}


COSINE_THRESHOLDS = {
    "FP32": {
        "min_cosine": 0.995,
        "max_mismatch_rate": TOLERANT_THRESHOLDS["FP32"]["max_mismatch_rate"],
        "max_p99_abs": TOLERANT_THRESHOLDS["FP32"]["max_p99_abs"],
        "max_p99_rel": TOLERANT_THRESHOLDS["FP32"]["max_p99_rel"],
        "min_norm_ratio": 0.98,
        "max_norm_ratio": 1.02,
    },
    "FP16": {
        "min_cosine": 0.99,
        "max_mismatch_rate": TOLERANT_THRESHOLDS["FP16"]["max_mismatch_rate"],
        "max_p99_abs": TOLERANT_THRESHOLDS["FP16"]["max_p99_abs"],
        "max_p99_rel": TOLERANT_THRESHOLDS["FP16"]["max_p99_rel"],
        "min_norm_ratio": 0.80,
        "max_norm_ratio": 1.25,
    },
    "BF16": {
        "min_cosine": 0.99,
        "max_mismatch_rate": TOLERANT_THRESHOLDS["BF16"]["max_mismatch_rate"],
        "max_p99_abs": TOLERANT_THRESHOLDS["BF16"]["max_p99_abs"],
        "max_p99_rel": TOLERANT_THRESHOLDS["BF16"]["max_p99_rel"],
        "min_norm_ratio": 0.75,
        "max_norm_ratio": 1.35,
    },
}


DEFAULT_THRESHOLDS_BY_POLICY = {
    POLICY_TOLERANT: TOLERANT_THRESHOLDS,
    POLICY_COSINE: COSINE_THRESHOLDS,
}


def _default_policy_name(category: str, op_name: str) -> str:
    if category == "reduction":
        if op_name in {"argmax", "max", "max_reduce"}:
            return POLICY_STRICT
        if op_name in {"cumsum", "sum_reduce"}:
            return POLICY_COSINE
    return CATEGORY_DEFAULT_POLICIES.get(category, POLICY_STRICT)


def _merge_thresholds(
    base_thresholds: Dict[str, Dict[str, float]],
    override_thresholds: Any,
) -> Dict[str, Dict[str, float]]:
    merged = deepcopy(base_thresholds)
    if not isinstance(override_thresholds, dict):
        return merged

    for dtype_name, dtype_thresholds in override_thresholds.items():
        if not isinstance(dtype_thresholds, dict):
            continue
        normalized_dtype = str(dtype_name).upper()
        current = dict(merged.get(normalized_dtype, {}))
        for key, value in dtype_thresholds.items():
            current[key] = value
        merged[normalized_dtype] = current
    return merged


def resolve_correctness_policy(operator_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective correctness policy for one operator."""
    config = operator_spec.get("config") or {}
    category = str(operator_spec.get("category") or "")
    op_name = str(operator_spec.get("op_name") or "")

    default_policy = _default_policy_name(category, op_name)
    configured_policy = config.get("policy")
    policy_name = str(configured_policy or default_policy)
    default_thresholds = DEFAULT_THRESHOLDS_BY_POLICY.get(policy_name, {})
    thresholds = _merge_thresholds(default_thresholds, config.get("thresholds"))
    source = "config" if configured_policy is not None else "category_default"

    return {
        "policy": policy_name,
        "category": category,
        "source": source,
        "thresholds": thresholds,
    }


def policy_name(correctness_policy: Dict[str, Any] | None) -> str:
    if not correctness_policy:
        return POLICY_STRICT
    return str(correctness_policy.get("policy") or POLICY_STRICT)


def is_tolerant_policy(correctness_policy: Dict[str, Any] | None) -> bool:
    return policy_name(correctness_policy) == POLICY_TOLERANT


def is_cosine_policy(correctness_policy: Dict[str, Any] | None) -> bool:
    return policy_name(correctness_policy) == POLICY_COSINE


def is_metric_policy(correctness_policy: Dict[str, Any] | None) -> bool:
    return is_tolerant_policy(correctness_policy) or is_cosine_policy(correctness_policy)


def thresholds_for_dtype(
    correctness_policy: Dict[str, Any] | None,
    dtype_name: str,
) -> Dict[str, float]:
    if not correctness_policy:
        return {}
    thresholds = correctness_policy.get("thresholds") or {}
    normalized_dtype = str(dtype_name).upper()
    configured = (
        thresholds.get(normalized_dtype)
        or thresholds.get(normalized_dtype.lower())
        or thresholds.get(normalized_dtype.capitalize())
    )
    if configured is not None:
        return dict(configured)
    return dict(thresholds.get("FP32") or {})
