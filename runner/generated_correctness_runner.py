"""Correctness checks for generated CUDA/SUPA code against PyTorch operators."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from operators.registry import (
    get_operator_spec,
    load_object,
    load_module,
)
from runner.correctness_policy import (
    is_cosine_policy,
    is_metric_policy,
    is_tolerant_policy,
    policy_name,
    resolve_correctness_policy,
    thresholds_for_dtype,
)
from runner.profiler_runner import (
    ensure_backend,
    profile_case,
    setup_backend_device,
)
from runner.random_shape_resolver import generate_random_seed, resolve_random_shapes


TOLERANCE_POLICY = {
    "fp32": {"rtol": 1.3e-6, "atol": 1e-5},
    "fp16": {"rtol": 0.001, "atol": 0.0001},
    "bf16": {"rtol": 0.016, "atol": 1e-3},
    "int8": {"max_abs_diff": 1},
}
BASELINE_COMPUTE_MODE_KEY = "_baseline_compute_mode"
BASELINE_COMPUTE_MODE_CORRECTNESS = "correctness"
BASELINE_COMPUTE_MODE_PERFORMANCE = "performance"
CORRECTNESS_BASELINE_COMPUTE_MODE_CONFIG_KEY = "correctness_baseline_compute_mode"
RANDOM_CORRECTNESS_CONFIG_KEY = "random_correctness"
DEFAULT_RANDOM_CORRECTNESS = {"max_failed_case_rate": 0.03}
CUDA_OOM_MAX_RETRIES = 3
COMPILE_SUBPROCESS_TIMEOUT_SECONDS = 1200
CASE_SUBPROCESS_TIMEOUT_SECONDS = 600
EVALUATION_RESULT_FIELD_ORDER = [
    "op_name",
    "shape_mode",
    "language",
    "devices",
    "run_id",
    "random",
    "perf_trials_per_shape",
    "generated_code",
    "reward_hacking",
    "compiled",
    "correctness",
    "cases",
    "status",
    "total_cases",
    "completed_cases",
    "current_case_index",
    "last_completed_case_index",
    "running_cases",
    "last_updated",
]


@contextmanager
def _disable_tf32_for_correctness():
    import torch

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _to_cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def _clone_input(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_input(item) for item in value)
    if isinstance(value, list):
        return [_clone_input(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_input(item) for key, item in value.items()}
    return value


def _seed_all(seed: int, language: str) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if str(language).lower() == "biren" and hasattr(torch, "supa"):
        supa_mod = getattr(torch, "supa")
        if hasattr(supa_mod, "manual_seed_all"):
            try:
                supa_mod.manual_seed_all(seed)
            except Exception:
                pass
    if str(language).lower() == "ascendc":
        try:
            import torch_npu

            torch_npu.npu.manual_seed_all(seed)
        except (ImportError, AttributeError):
            pass


def _is_cuda_oom_error(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return True
    except Exception:
        pass

    message = str(error).lower()
    return (
        "cuda out of memory" in message
        or "supa out of memory" in message
        or "npu out of memory" in message
        or "outofmemoryerror" in message
    )


def _cleanup_after_cuda_oom() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        if hasattr(torch, "supa"):
            try:
                torch.supa.empty_cache()
            except Exception:
                pass
        try:
            import torch_npu

            torch_npu.npu.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass
    except Exception:
        pass


def _dtype_name(tensor) -> str:
    import torch

    if tensor.dtype == torch.float32:
        return "FP32"
    if tensor.dtype == torch.float16:
        return "FP16"
    if tensor.dtype == torch.bfloat16:
        return "BF16"
    if tensor.dtype in (torch.int8, torch.uint8):
        return "INT8"
    return str(tensor.dtype).replace("torch.", "").upper()


def _shape_with_baseline_compute_mode(
    shape: Dict[str, Any],
    baseline_compute_mode: str,
) -> Dict[str, Any]:
    case_shape = dict(shape)
    case_shape[BASELINE_COMPUTE_MODE_KEY] = baseline_compute_mode
    return case_shape


def _resolve_correctness_baseline_compute_mode(operator_spec: Dict[str, Any]) -> str:
    config = operator_spec.get("config") or {}
    mode = str(
        config.get(
            CORRECTNESS_BASELINE_COMPUTE_MODE_CONFIG_KEY,
            BASELINE_COMPUTE_MODE_CORRECTNESS,
        )
    )
    if mode not in {BASELINE_COMPUTE_MODE_CORRECTNESS, BASELINE_COMPUTE_MODE_PERFORMANCE}:
        op_name = operator_spec.get("op_name", "<unknown>")
        raise ValueError(
            f"{op_name} has invalid {CORRECTNESS_BASELINE_COMPUTE_MODE_CONFIG_KEY}: "
            f"{mode!r}. Expected 'correctness' or 'performance'."
        )
    return mode


def _resolve_random_correctness_config(operator_spec: Dict[str, Any]) -> Dict[str, float]:
    config = operator_spec.get("config") or {}
    random_correctness = config.get(RANDOM_CORRECTNESS_CONFIG_KEY) or {}
    if not isinstance(random_correctness, dict):
        op_name = operator_spec.get("op_name", "<unknown>")
        raise ValueError(f"{op_name} has invalid {RANDOM_CORRECTNESS_CONFIG_KEY}: expected object.")

    raw_rate = random_correctness.get(
        "max_failed_case_rate",
        DEFAULT_RANDOM_CORRECTNESS["max_failed_case_rate"],
    )
    try:
        max_failed_case_rate = float(raw_rate)
    except (TypeError, ValueError) as exc:
        op_name = operator_spec.get("op_name", "<unknown>")
        raise ValueError(
            f"{op_name} has invalid random_correctness.max_failed_case_rate: {raw_rate!r}."
        ) from exc
    if not 0.0 <= max_failed_case_rate <= 1.0:
        op_name = operator_spec.get("op_name", "<unknown>")
        raise ValueError(
            f"{op_name} has invalid random_correctness.max_failed_case_rate: "
            f"{max_failed_case_rate}. Expected a value in [0.0, 1.0]."
        )
    return {"max_failed_case_rate": max_failed_case_rate}


def _random_case_summary(
    *,
    shape_mode: str,
    total_cases: int,
    failed_cases: int,
    random_correctness_config: Dict[str, float],
) -> Dict[str, Any]:
    max_failed_case_rate = float(random_correctness_config["max_failed_case_rate"])
    if total_cases <= 0:
        failed_case_rate = 1.0
    else:
        failed_case_rate = failed_cases / total_cases
    within_tolerance = (
        shape_mode == "random"
        and total_cases > 0
        and failed_case_rate <= max_failed_case_rate
    )
    return {
        "total_cases": total_cases,
        "failed_cases": failed_cases,
        "failed_case_rate": round(failed_case_rate, 4),
        "max_failed_case_rate": max_failed_case_rate,
        "within_tolerance": within_tolerance,
    }


def _dtype_policy(tensor, tolerance_policy: Optional[Dict[str, Any]] = None) -> Dict[str, float | int]:
    import torch

    dtype_name = _dtype_name(tensor)
    if tolerance_policy:
        configured_policy = (
            tolerance_policy.get(dtype_name)
            or tolerance_policy.get(dtype_name.lower())
            or tolerance_policy.get(dtype_name.upper())
        )
        if configured_policy is not None:
            return configured_policy

    if tensor.dtype in (torch.int8, torch.uint8):
        return TOLERANCE_POLICY["int8"]
    if tensor.dtype == torch.float32:
        return TOLERANCE_POLICY["fp32"]
    if tensor.dtype == torch.float16:
        return TOLERANCE_POLICY["fp16"]
    if tensor.dtype == torch.bfloat16:
        return TOLERANCE_POLICY["bf16"]
    if torch.is_floating_point(tensor):
        return TOLERANCE_POLICY["fp32"]
    return {"max_abs_diff": 0}


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _sample_flat_values(values, max_samples: int = 1_000_000):
    import torch

    values = values.reshape(-1)
    numel = values.numel()
    if numel <= max_samples:
        return values

    sample_indices = torch.arange(max_samples, device=values.device, dtype=torch.int64)
    sample_indices = sample_indices * (numel - 1) // (max_samples - 1)
    return values[sample_indices]


def _cosine_similarity_meta(generated, reference, max_samples: int = 1_000_000) -> Optional[float]:
    import torch

    if generated.numel() == 0 or not torch.is_floating_point(reference):
        return None

    gen_flat = _sample_flat_values(generated, max_samples)
    ref_flat = _sample_flat_values(reference, max_samples)

    gen_flat = gen_flat.to(torch.float64)
    ref_flat = ref_flat.to(torch.float64)
    gen_norm = torch.linalg.vector_norm(gen_flat)
    ref_norm = torch.linalg.vector_norm(ref_flat)
    gen_zero = bool(gen_norm.item() == 0.0)
    ref_zero = bool(ref_norm.item() == 0.0)
    if gen_zero and ref_zero:
        return 1.0
    if gen_zero or ref_zero:
        return 0.0
    cosine = torch.dot(gen_flat, ref_flat) / (gen_norm * ref_norm)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    return round(float(cosine.item()), 4)


def _norm_ratio_meta(generated, reference, max_samples: int = 1_000_000) -> Optional[float]:
    import torch

    if generated.numel() == 0 or not torch.is_floating_point(reference):
        return None

    gen_flat = _sample_flat_values(generated, max_samples).to(torch.float64)
    ref_flat = _sample_flat_values(reference, max_samples).to(torch.float64)
    gen_norm = torch.linalg.vector_norm(gen_flat)
    ref_norm = torch.linalg.vector_norm(ref_flat)
    gen_zero = bool(gen_norm.item() == 0.0)
    ref_zero = bool(ref_norm.item() == 0.0)
    if gen_zero and ref_zero:
        return 1.0
    if gen_zero:
        return 0.0
    if ref_zero:
        return float("inf")
    return round(float((gen_norm / ref_norm).item()), 4)


def _p99_meta(values, max_samples: int = 1_000_000) -> Optional[float]:
    import torch

    if values.numel() == 0:
        return None
    values = _sample_flat_values(values, max_samples).to(torch.float32)
    return float(torch.quantile(values, 0.99).item())


def _mean_meta(values, max_samples: int = 1_000_000) -> Optional[float]:
    import torch

    if values.numel() == 0:
        return None
    values = _sample_flat_values(values, max_samples).to(torch.float32)
    return float(values.mean().item())


def _finite_pattern_pass(generated, reference) -> bool:
    import torch

    if not torch.is_floating_point(reference):
        return True

    generated_finite = torch.isfinite(generated)
    reference_finite = torch.isfinite(reference)
    if not torch.equal(generated_finite, reference_finite):
        return False

    nonfinite_mask = ~reference_finite
    if not bool(nonfinite_mask.any().item()):
        return True

    generated_nonfinite = generated[nonfinite_mask]
    reference_nonfinite = reference[nonfinite_mask]
    same_nan = torch.isnan(generated_nonfinite) & torch.isnan(reference_nonfinite)
    same_inf = (
        torch.isinf(generated_nonfinite)
        & torch.isinf(reference_nonfinite)
        & (generated_nonfinite == reference_nonfinite)
    )
    return bool((same_nan | same_inf).all().item())


def _threshold_pass(value: Any, threshold: Any) -> bool:
    import math

    if value is None or threshold is None:
        return False
    value_float = float(value)
    threshold_float = float(threshold)
    if not math.isfinite(value_float) or not math.isfinite(threshold_float):
        return False
    return value_float <= threshold_float


def _cosine_pass(value: Any, threshold: Any) -> bool:
    import math

    if value is None or threshold is None:
        return False
    value_float = float(value)
    threshold_float = float(threshold)
    if not math.isfinite(value_float) or not math.isfinite(threshold_float):
        return False
    return value_float >= threshold_float


def _range_pass(value: Any, min_value: Any, max_value: Any) -> bool:
    import math

    if value is None or min_value is None or max_value is None:
        return False
    value_float = float(value)
    min_float = float(min_value)
    max_float = float(max_value)
    if (
        not math.isfinite(value_float)
        or not math.isfinite(min_float)
        or not math.isfinite(max_float)
    ):
        return False
    return min_float <= value_float <= max_float


def _near_zero_abs_policy_pass(
    reference,
    abs_diff,
    mismatch_mask,
    meta: Dict[str, Any],
    correctness_policy: Optional[Dict[str, Any]],
) -> bool:
    import torch

    meta["near_zero_abs_pass"] = False
    if not torch.is_floating_point(reference):
        return False
    if reference.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not is_tolerant_policy(correctness_policy):
        return False

    thresholds = thresholds_for_dtype(correctness_policy, _dtype_name(reference))
    near_zero_abs = thresholds.get("near_zero_abs")
    max_bad_abs = thresholds.get("max_near_zero_bad_abs")
    if near_zero_abs is None or max_bad_abs is None:
        return False

    reference_abs = reference.to(torch.float32).abs()
    near_zero_mask = reference_abs <= float(near_zero_abs)
    far_mask = ~near_zero_mask
    near_zero_count = int(near_zero_mask.sum().item())
    far_count = int(far_mask.sum().item())
    numel = int(reference.numel())
    far_mismatch_count = int((mismatch_mask & far_mask).sum().item())

    meta["near_zero_abs"] = float(near_zero_abs)
    meta["near_zero_rate"] = round((near_zero_count / numel) if numel else 0.0, 4)
    meta["far_mismatch_rate"] = round(
        (far_mismatch_count / far_count) if far_count else 0.0,
        4,
    )

    if near_zero_count == 0:
        meta["near_zero_p99_abs"] = None
        meta["near_zero_mean_abs"] = None
        meta["near_zero_bad_rate"] = 0.0
        return False

    near_zero_abs_diff = abs_diff[near_zero_mask]
    near_zero_bad_mask = near_zero_abs_diff > float(max_bad_abs)
    near_zero_bad_count = int(near_zero_bad_mask.sum().item())
    meta["near_zero_p99_abs"] = _p99_meta(near_zero_abs_diff)
    meta["near_zero_mean_abs"] = _mean_meta(near_zero_abs_diff)
    meta["near_zero_bad_rate"] = round(near_zero_bad_count / near_zero_count, 4)

    meta["near_zero_abs_pass"] = bool(
        meta.get("finite_pass")
        and far_mismatch_count == 0
        and _threshold_pass(
            meta.get("near_zero_p99_abs"),
            thresholds.get("max_near_zero_p99_abs"),
        )
        and _threshold_pass(
            meta.get("near_zero_mean_abs"),
            thresholds.get("max_near_zero_mean_abs"),
        )
        and _threshold_pass(
            meta.get("near_zero_bad_rate"),
            thresholds.get("max_near_zero_bad_rate"),
        )
    )
    return bool(meta["near_zero_abs_pass"])


def _evaluate_tensor_policy(
    reference,
    meta: Dict[str, Any],
    correctness_policy: Optional[Dict[str, Any]],
) -> bool:
    import torch

    if meta.get("allclose_pass"):
        return True
    if not torch.is_floating_point(reference):
        return False
    if is_cosine_policy(correctness_policy):
        thresholds = thresholds_for_dtype(correctness_policy, _dtype_name(reference))
        cosine_ok = _cosine_pass(meta.get("cosine_similarity"), thresholds.get("min_cosine"))
        mismatch_ok = _threshold_pass(
            meta.get("mismatch_rate"),
            thresholds.get("max_mismatch_rate"),
        )
        p99_abs_ok = _threshold_pass(meta.get("p99_abs"), thresholds.get("max_p99_abs"))
        p99_rel_ok = _threshold_pass(meta.get("p99_rel"), thresholds.get("max_p99_rel"))
        norm_ratio_ok = _range_pass(
            meta.get("norm_ratio"),
            thresholds.get("min_norm_ratio"),
            thresholds.get("max_norm_ratio"),
        )
        finite_pass = bool(meta.get("finite_pass"))
        normal_metric_pass = bool(
            finite_pass
            and cosine_ok
            and mismatch_ok
            and (p99_abs_ok or p99_rel_ok)
        )
        cosine_metric_pass = bool(finite_pass and cosine_ok and norm_ratio_ok)
        meta["normal_metric_pass"] = normal_metric_pass
        meta["norm_ratio_pass"] = norm_ratio_ok
        meta["cosine_metric_pass"] = cosine_metric_pass
        return bool(normal_metric_pass or cosine_metric_pass)
    if not is_tolerant_policy(correctness_policy):
        return False

    thresholds = thresholds_for_dtype(correctness_policy, _dtype_name(reference))
    cosine_ok = _cosine_pass(meta.get("cosine_similarity"), thresholds.get("min_cosine"))
    mismatch_ok = _threshold_pass(
        meta.get("mismatch_rate"),
        thresholds.get("max_mismatch_rate"),
    )
    p99_abs_ok = _threshold_pass(meta.get("p99_abs"), thresholds.get("max_p99_abs"))
    p99_rel_ok = _threshold_pass(meta.get("p99_rel"), thresholds.get("max_p99_rel"))
    normal_metric_pass = bool(
        meta.get("finite_pass")
        and cosine_ok
        and mismatch_ok
        and (p99_abs_ok or p99_rel_ok)
    )
    meta["normal_metric_pass"] = normal_metric_pass
    return bool(normal_metric_pass)


def _correctness_criteria(
    tolerance_policy: Dict[str, Any],
    correctness_policy: Dict[str, Any],
    baseline_compute_mode: str,
    random_correctness_config: Dict[str, float],
    shape_mode: str,
) -> Dict[str, Any]:
    metric_enabled = is_metric_policy(correctness_policy)
    criteria = {
        "baseline_compute_mode": baseline_compute_mode,
        "policy": {
            "name": policy_name(correctness_policy),
            "category": correctness_policy.get("category"),
            "source": correctness_policy.get("source"),
        },
        "allclose": {
            "enabled": True,
            "tolerances": tolerance_policy,
        },
        "metric_fallback": {
            "enabled": metric_enabled,
        },
        "random_correctness": {
            "enabled": shape_mode == "random",
            "max_failed_case_rate": random_correctness_config["max_failed_case_rate"],
        },
    }
    if metric_enabled and is_cosine_policy(correctness_policy):
        criteria["metric_fallback"].update(
            {
                "pass_rule": (
                    "allclose_pass OR normal_metric_pass OR cosine_metric_pass"
                ),
                "tolerant_metric_pass_rule": (
                    "finite_pass AND "
                    "cosine_similarity >= min_cosine AND "
                    "mismatch_rate <= max_mismatch_rate AND "
                    "(p99_abs <= max_p99_abs OR p99_rel <= max_p99_rel)"
                ),
                "cosine_metric_pass_rule": (
                    "finite_pass AND "
                    "cosine_similarity >= min_cosine AND "
                    "min_norm_ratio <= norm_ratio <= max_norm_ratio"
                ),
                "thresholds": correctness_policy.get("thresholds") or {},
            }
        )
    elif metric_enabled:
        criteria["metric_fallback"].update(
            {
                "pass_rule": "allclose_pass OR normal_metric_pass",
                "metric_pass_rule": (
                    "finite_pass AND "
                    "cosine_similarity >= min_cosine AND "
                    "mismatch_rate <= max_mismatch_rate AND "
                    "(p99_abs <= max_p99_abs OR p99_rel <= max_p99_rel)"
                ),
                "thresholds": correctness_policy.get("thresholds") or {},
            }
        )
    return criteria


def _compare_tensors(
    generated,
    reference,
    path: str,
    tolerance_policy: Optional[Dict[str, Any]] = None,
    correctness_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    import torch

    active_policy_name = policy_name(correctness_policy)
    meta: Dict[str, Any] = {
        "path": path,
        "correctness_policy": active_policy_name,
    }

    if generated.shape != reference.shape:
        meta["allclose_pass"] = False
        meta["metric_pass"] = False
        meta["policy_pass"] = False
        meta["mismatch_rate"] = 1.0
        meta["p99_abs"] = None
        meta["p99_rel"] = None
        return (
            False,
            [f"{path}.shape mismatch: expected {tuple(reference.shape)}, got {tuple(generated.shape)}"],
            meta,
        )
    if generated.dtype != reference.dtype:
        meta["allclose_pass"] = False
        meta["metric_pass"] = False
        meta["policy_pass"] = False
        meta["mismatch_rate"] = 1.0
        meta["p99_abs"] = None
        meta["p99_rel"] = None
        return (
            False,
            [f"{path}.dtype mismatch: expected {reference.dtype}, got {generated.dtype}"],
            meta,
        )
    meta["cosine_similarity"] = _cosine_similarity_meta(generated, reference)
    meta["norm_ratio"] = _norm_ratio_meta(generated, reference)

    policy = _dtype_policy(reference, tolerance_policy)
    if "max_abs_diff" in policy:
        gen_cmp = generated.to(torch.int32)
        ref_cmp = reference.to(torch.int32)
        abs_diff = (gen_cmp - ref_cmp).abs()
        mismatch_mask = abs_diff > int(policy["max_abs_diff"])
        tolerance = {"max_abs_diff": int(policy["max_abs_diff"])}
    else:
        gen_cmp = generated.to(torch.float32)
        ref_cmp = reference.to(torch.float32)
        abs_diff = (gen_cmp - ref_cmp).abs()
        rtol = float(policy["rtol"])
        atol = float(policy["atol"])
        mismatch_mask = ~torch.isclose(ref_cmp, gen_cmp, rtol=rtol, atol=atol)
        tolerance = {"rtol": rtol, "atol": atol}

    rel_diff = abs_diff.to(torch.float32) / (reference.to(torch.float32).abs() + 1e-12)
    mismatch_count = int(mismatch_mask.sum().item())
    numel = int(reference.numel())
    allclose_pass = bool(mismatch_count == 0)
    meta["mismatch_rate"] = round((mismatch_count / numel) if numel else 0.0, 4)
    meta["p99_abs"] = _p99_meta(abs_diff)
    meta["p99_rel"] = _p99_meta(rel_diff)
    meta["allclose_pass"] = allclose_pass
    meta["normal_metric_pass"] = False
    meta["cosine_metric_pass"] = False
    meta["norm_ratio_pass"] = False
    if torch.is_floating_point(reference):
        meta["finite_pass"] = _finite_pattern_pass(generated, reference)
        _near_zero_abs_policy_pass(
            reference,
            abs_diff,
            mismatch_mask,
            meta,
            correctness_policy,
        )
    meta["metric_pass"] = (
        False if allclose_pass else _evaluate_tensor_policy(reference, meta, correctness_policy)
    )
    meta["policy_pass"] = bool(allclose_pass or meta["metric_pass"])
    if generated.numel() == 0 or mismatch_count == 0:
        return True, [], meta

    mismatch_abs = abs_diff[mismatch_mask]
    mismatch_rel = rel_diff[mismatch_mask]
    max_abs = mismatch_abs.max().item()
    max_rel = mismatch_rel.max().item()
    meta["max_abs"] = float(max_abs)
    meta["max_rel"] = float(max_rel)

    first_index = None
    if mismatch_mask.dim() == 0:
        first_index = "()"
    else:
        indices = torch.nonzero(mismatch_mask, as_tuple=False)
        if indices.numel() > 0:
            first_index = tuple(int(v) for v in indices[0].tolist())
    meta["first_mismatch_index"] = first_index

    detail = (
        f"{path}: policy={active_policy_name}, policy_pass={meta['policy_pass']}, "
        f"allclose_pass={meta['allclose_pass']}, metric_pass={meta['metric_pass']}, "
        f"normal_metric_pass={meta.get('normal_metric_pass')}, "
        f"cosine_metric_pass={meta.get('cosine_metric_pass')}, "
        f"near_zero_abs_pass={meta.get('near_zero_abs_pass')}, "
        f"{mismatch_count}/{int(reference.numel())} elements mismatched, "
        f"max_abs={_format_number(max_abs)}, max_rel={_format_number(max_rel)}, "
        f"p99_abs={_format_number(meta['p99_abs'])}, p99_rel={_format_number(meta['p99_rel'])}, "
        f"cosine_similarity={_format_number(meta.get('cosine_similarity'))}, "
        f"norm_ratio={_format_number(meta.get('norm_ratio'))}, "
        f"first_mismatch_index={first_index}, tolerance={tolerance}"
    )
    return bool(meta["policy_pass"]), ([] if meta["policy_pass"] else [detail]), meta


def _compare_outputs(
    generated: Any,
    reference: Any,
    path: str = "output",
    tolerance_policy: Optional[Dict[str, Any]] = None,
    correctness_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    import torch

    if isinstance(generated, torch.Tensor) and isinstance(reference, torch.Tensor):
        ok, details, meta = _compare_tensors(
            generated,
            reference,
            path,
            tolerance_policy,
            correctness_policy,
        )
        return ok, details, [meta]

    if isinstance(generated, (list, tuple)) and isinstance(reference, (list, tuple)):
        if len(generated) != len(reference):
            return False, [f"{path} length mismatch: expected {len(reference)}, got {len(generated)}"], []
        all_ok = True
        all_details: List[str] = []
        all_meta: List[Dict[str, Any]] = []
        for index, (gen_item, ref_item) in enumerate(zip(generated, reference)):
            ok, details, meta = _compare_outputs(
                gen_item,
                ref_item,
                f"{path}[{index}]",
                tolerance_policy,
                correctness_policy,
            )
            all_ok = all_ok and ok
            all_details.extend(details)
            all_meta.extend(meta)
        return all_ok, all_details, all_meta

    if generated == reference:
        return True, [], []
    return (
        False,
        [f"{path} type/value mismatch: expected {type(reference).__name__}, got {type(generated).__name__}"],
        [],
    )


def _extract_first_code(text: str) -> str:
    """Accept either raw Python text or a fenced Markdown code block."""
    match = re.search(r"```(.*?)```", text.strip(), re.DOTALL)
    if not match:
        return text
    code = match.group(1).strip()
    for prefix in ("python", "cpp"):
        if code.startswith(prefix):
            return code[len(prefix) :].strip()
    return code


def _resolve_shapes(
    shape_module_name: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_seed: Optional[int],
    random_count: int,
    random_max_attempts: int,
    validate_shape_fn,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    shape_module = load_module(shape_module_name)
    single_shapes = getattr(shape_module, "SINGLE_SHAPES")
    shape_families = getattr(shape_module, "SHAPE_FAMILIES")

    if shape_mode == "single":
        return [single_shapes[shape_name]], None
    if shape_mode == "random":
        resolved_seed = generate_random_seed() if random_seed is None else int(random_seed)
        shapes = resolve_random_shapes(
            shape_module_name,
            seed=resolved_seed,
            count=random_count,
            max_attempts=random_max_attempts,
            validate_shape_fn=validate_shape_fn,
        )
        return shapes, {
            "seed": resolved_seed,
            "count": random_count,
            "max_attempts": random_max_attempts,
        }
    if shape_family == "all":
        shapes: List[Dict[str, Any]] = []
        for family_shapes in shape_families.values():
            shapes.extend(list(family_shapes))
        return shapes, None
    return list(shape_families[shape_family]), None


def _operator_tolerance_policy(operator_spec: Dict[str, Any]) -> Dict[str, Any]:
    config = operator_spec.get("config") or {}
    return config.get("tolerances") or TOLERANCE_POLICY


def _compile_generated_code(backend: Any, generated_code: str, op_name: str) -> Tuple[bool, Optional[str]]:
    backend.context = {}
    return backend.compile(_extract_first_code(generated_code), op_name)


def _profile_model_case(
    *,
    language: str,
    device_id: int,
    op_name: str,
    case_name: str,
    shape: Dict[str, Any],
    model_class: type,
    generate_inputs_fn,
    perf_trials: int,
) -> Dict[str, Any]:
    """Profile one model class for a verify case."""
    oom_retries = 0
    last_error: Optional[BaseException] = None
    for attempt in range(CUDA_OOM_MAX_RETRIES + 1):
        try:
            result = profile_case(
                language=language,
                device_id=device_id,
                op_name=op_name,
                case_name=case_name,
                shape=shape,
                model_class=model_class,
                generate_inputs_fn=generate_inputs_fn,
                perf_trials=perf_trials,
                use_torch_compile=False,
            )
            if oom_retries:
                result["oom_retries"] = oom_retries
            return result
        except Exception as error:  # pylint: disable=broad-except
            last_error = error
            if not _is_cuda_oom_error(error) or attempt >= CUDA_OOM_MAX_RETRIES:
                if _is_cuda_oom_error(error):
                    _cleanup_after_cuda_oom()
                break
            oom_retries += 1
            _cleanup_after_cuda_oom()

    payload = {
        "performance": None,
        "profiling": None,
        "performance_error_message": str(last_error),
    }
    if oom_retries:
        payload["oom_retries"] = oom_retries
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subprocess_stderr_summary(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    if stderr:
        return stderr[-4000:]
    if stdout:
        return stdout[-4000:]
    return ""


def _case_subprocess_failure_result(
    *,
    raw_shape: Dict[str, Any],
    case_index: int,
    error_type: str,
    error_message: str,
    device_id: Optional[int] = None,
) -> Dict[str, Any]:
    failure = {
        "case_index": case_index,
        "trial": None,
        "details": [error_message],
    }
    case_result = {
        "case_index": case_index,
        "shape": raw_shape,
        "success": False,
        "failed_trials": [failure],
        "performance": {
            "generated": None,
            "reference": None,
        },
        "profiling": {
            "generated": None,
            "reference": None,
        },
        "output_meta": [],
        "error_type": error_type,
        "error_message": error_message,
    }
    if device_id is not None:
        case_result["device_id"] = device_id
    return {
        "case_result": case_result,
        "failed_trials": [failure],
        "error_type": error_type,
        "error_message": error_message,
    }


def _case_error_text(case_result: Dict[str, Any]) -> str:
    """Flatten case diagnostics for backend-specific failure classification."""
    messages = [
        str(case_result.get("error_type") or ""),
        str(case_result.get("error_message") or ""),
    ]
    for failure in case_result.get("failed_trials") or []:
        if isinstance(failure, dict):
            messages.extend(str(detail) for detail in failure.get("details") or [])

    performance_error = case_result.get("performance_error_message")
    if isinstance(performance_error, dict):
        messages.extend(str(value) for value in performance_error.values() if value)
    elif performance_error:
        messages.append(str(performance_error))
    return "\n".join(messages).lower()


def _is_biren_case_compile_failure(case_result: Dict[str, Any]) -> bool:
    diagnostics = _case_error_text(case_result)
    direct_compile_failure = any(
        marker in diagnostics
        for marker in (
            "casecompilefailed",
            "error building extension",
            "build command failed",
            "generated code compilation failed",
            "ninja: build stopped",
            "compilation terminated",
        )
    )
    compiler_fatal_error = "fatal error:" in diagnostics and any(
        marker in diagnostics
        for marker in (
            "brcc",
            "kernel.su",
            "main.cpp",
            "file not found",
            "no such file or directory",
            "compiling for device",
        )
    )
    return direct_compile_failure or compiler_fatal_error


def _worker_subprocess_timeout_seconds(request: Dict[str, Any]) -> int:
    if request.get("mode") == "compile":
        return COMPILE_SUBPROCESS_TIMEOUT_SECONDS
    return CASE_SUBPROCESS_TIMEOUT_SECONDS


def _cleanup_torch_extensions_dir(path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    try:
        shutil.rmtree(Path(str(path_value)), ignore_errors=True)
    except Exception as exc:  # pylint: disable=broad-except
        return f" Failed to clean torch extensions dir {path_value}: {exc}"
    return None


def _worker_timeout_result(request: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    mode = request.get("mode")
    cleanup_message = _cleanup_torch_extensions_dir(request.get("torch_extensions_dir"))
    if mode == "case":
        message = f"Case worker timed out after {timeout_seconds} seconds."
        if cleanup_message:
            message += cleanup_message
        return _case_subprocess_failure_result(
            raw_shape=dict(request.get("raw_shape") or {}),
            case_index=int(request.get("case_index", -1)),
            error_type="CaseTimeout",
            error_message=message,
            device_id=int(request.get("device_id", -1)),
        )

    message = f"Compile worker timed out after {timeout_seconds} seconds."
    if cleanup_message:
        message += cleanup_message
    return {
        "mode": mode,
        "device": None,
        "compiled": False,
        "compile_info": message,
        "has_model_new": False,
        "error_type": "CompileTimeout",
        "error_message": message,
    }


def _run_worker_subprocess(request: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="op_eval_case_") as temp_dir:
        temp_path = Path(temp_dir)
        request_path = temp_path / "request.json"
        result_path = temp_path / "result.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        env = os.environ.copy()
        if str(request.get("language", "")).lower() == "biren":
            env["SUPA_VISIBLE_DEVICES"] = str(int(request["device_id"]))
        torch_extensions_dir = request.get("torch_extensions_dir")
        if torch_extensions_dir:
            env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)

        timeout_seconds = _worker_subprocess_timeout_seconds(request)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "runner.generated_case_worker",
                    "--request-json",
                    str(request_path),
                    "--result-json",
                    str(result_path),
                ],
                cwd=str(_repo_root()),
                text=True,
                capture_output=True,
                check=False,
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return _worker_timeout_result(request, timeout_seconds)

        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if request.get("mode") == "compile":
                    payload["stdout"] = completed.stdout or ""
                    payload["stderr"] = completed.stderr or ""
                return payload
            except json.JSONDecodeError as exc:
                return {
                    "error_type": "WorkerResultDecodeError",
                    "error_message": f"Could not parse worker result JSON: {exc}",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }

        details = _subprocess_stderr_summary(completed)
        message = (
            f"Case worker exited with return code {completed.returncode} "
            "without writing result JSON."
        )
        if details:
            message = f"{message}\n{details}"
        return {
            "error_type": "CaseSubprocessFailed",
            "error_message": message,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


def _run_compile_subprocess(
    *,
    op_name: str,
    language: str,
    generated_path: str,
    device_id: int,
    torch_extensions_dir: Optional[str] = None,
) -> Dict[str, Any]:
    request = {
        "mode": "compile",
        "op_name": op_name,
        "language": language,
        "generated_path": generated_path,
        "device_id": device_id,
    }
    if torch_extensions_dir:
        request["torch_extensions_dir"] = torch_extensions_dir
    return _run_worker_subprocess(request)


def compile_generated_operator(
    *,
    op_name: str,
    language: str,
    generated_path: str,
    device_id: int,
    torch_extensions_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile one generated module without instantiating or running ModelNew."""
    generated_file = Path(generated_path)
    if not generated_file.exists():
        message = f"Generated code file not found: {generated_file}"
        return {
            "op_name": op_name,
            "language": language,
            "device_id": device_id,
            "device": None,
            "generated_code": str(generated_file),
            "compiled": False,
            "model_new_found": False,
            "success": False,
            "compile_info": message,
            "stdout": "",
            "stderr": "",
            "error_type": "GeneratedFileNotFound",
            "error_message": message,
        }

    worker_result = _run_compile_subprocess(
        op_name=op_name,
        language=language,
        generated_path=str(generated_file),
        device_id=device_id,
        torch_extensions_dir=torch_extensions_dir,
    )
    compiled = bool(worker_result.get("compiled"))
    model_new_found = bool(worker_result.get("has_model_new"))
    success = compiled and model_new_found
    compile_info = worker_result.get("compile_info")
    error_type = worker_result.get("error_type")
    error_message = worker_result.get("error_message")
    if compiled and not model_new_found:
        error_type = error_type or "ModelNewMissing"
        error_message = error_message or "Generated code must define ModelNew."
    elif not compiled:
        error_type = error_type or "CompilationFailed"
        error_message = error_message or compile_info or "Generated code compilation failed."

    return {
        "op_name": op_name,
        "language": language,
        "device_id": device_id,
        "device": worker_result.get("device"),
        "generated_code": str(generated_file),
        "compiled": compiled,
        "model_new_found": model_new_found,
        "success": success,
        "compile_info": compile_info,
        "stdout": worker_result.get("stdout") or "",
        "stderr": worker_result.get("stderr") or "",
        "error_type": error_type,
        "error_message": error_message,
    }


def _run_case_subprocess(
    *,
    op_name: str,
    language: str,
    generated_path: str,
    device_id: int,
    case_index: int,
    raw_shape: Dict[str, Any],
    trials: int,
    perf_trials: int,
    correctness_baseline_compute_mode: str,
    torch_extensions_dir: Optional[str] = None,
) -> Dict[str, Any]:
    request = {
        "mode": "case",
        "op_name": op_name,
        "language": language,
        "generated_path": generated_path,
        "device_id": device_id,
        "case_index": case_index,
        "raw_shape": raw_shape,
        "trials": trials,
        "perf_trials": perf_trials,
        "correctness_baseline_compute_mode": correctness_baseline_compute_mode,
    }
    if torch_extensions_dir:
        request["torch_extensions_dir"] = torch_extensions_dir
    payload = _run_worker_subprocess(request)
    if payload.get("case_result"):
        return payload
    return _case_subprocess_failure_result(
        raw_shape=raw_shape,
        case_index=case_index,
        error_type=str(payload.get("error_type") or "CaseSubprocessFailed"),
        error_message=str(payload.get("error_message") or "Case subprocess failed."),
        device_id=device_id,
    )


def _run_correctness_trial_once(
    *,
    language: str,
    correctness_shape: Dict[str, Any],
    generated_model_class: type,
    reference_model_class: type,
    generate_inputs_fn,
    backend: Any,
    device: Any,
    input_seed: int,
    param_seed: int,
    tolerance_policy: Dict[str, Any],
    correctness_policy: Dict[str, Any],
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    import torch

    _seed_all(input_seed, language)
    raw_inputs = _as_list(generate_inputs_fn(correctness_shape))
    device_inputs = [_to_device(item, device) for item in raw_inputs]
    generated_inputs = [_clone_input(item) for item in device_inputs]
    reference_inputs = [_clone_input(item) for item in device_inputs]

    with torch.no_grad(), _disable_tf32_for_correctness():
        _seed_all(param_seed, language)
        generated_model = generated_model_class(correctness_shape).to(device).eval()
        _seed_all(param_seed, language)
        reference_model = reference_model_class(correctness_shape).to(device).eval()
        generated_output = generated_model(*generated_inputs)
        if hasattr(backend, "synchronize"):
            backend.synchronize(device=device)
        reference_output = reference_model(*reference_inputs)
        if hasattr(backend, "synchronize"):
            backend.synchronize(device=device)

    generated_cpu = _to_cpu(generated_output)
    reference_cpu = _to_cpu(reference_output)
    return _compare_outputs(
        generated_cpu,
        reference_cpu,
        tolerance_policy=tolerance_policy,
        correctness_policy=correctness_policy,
    )


def _run_generated_case(
    *,
    op_name: str,
    language: str,
    device_id: int,
    case_index: int,
    raw_shape: Dict[str, Any],
    generated_model_class: type,
    reference_model_class: type,
    generate_inputs_fn,
    backend: Any,
    device: Any,
    trials: int,
    perf_trials: int,
    tolerance_policy: Dict[str, Any],
    correctness_policy: Dict[str, Any],
    correctness_baseline_compute_mode: str,
) -> Dict[str, Any]:
    correctness_shape = _shape_with_baseline_compute_mode(
        raw_shape,
        correctness_baseline_compute_mode,
    )
    performance_shape = _shape_with_baseline_compute_mode(
        raw_shape,
        BASELINE_COMPUTE_MODE_PERFORMANCE,
    )
    case_ok = True
    case_failures: List[Dict[str, Any]] = []
    failed_trials: List[Dict[str, Any]] = []
    output_meta: List[Dict[str, Any]] = []
    current_trial: Optional[int] = None
    correctness_error_message: Optional[str] = None
    correctness_oom_retries = 0

    try:
        for trial in range(trials):
            current_trial = trial
            input_seed = 1024 + case_index * 1000 + trial
            param_seed = input_seed + 10_000_000
            last_oom_error: Optional[BaseException] = None
            for attempt in range(CUDA_OOM_MAX_RETRIES + 1):
                try:
                    trial_ok, details, meta = _run_correctness_trial_once(
                        language=language,
                        correctness_shape=correctness_shape,
                        generated_model_class=generated_model_class,
                        reference_model_class=reference_model_class,
                        generate_inputs_fn=generate_inputs_fn,
                        backend=backend,
                        device=device,
                        input_seed=input_seed,
                        param_seed=param_seed,
                        tolerance_policy=tolerance_policy,
                        correctness_policy=correctness_policy,
                    )
                    break
                except Exception as error:  # pylint: disable=broad-except
                    if not _is_cuda_oom_error(error):
                        raise
                    _cleanup_after_cuda_oom()
                    if attempt >= CUDA_OOM_MAX_RETRIES:
                        raise
                    last_oom_error = error
                    correctness_oom_retries += 1
            else:
                raise RuntimeError(str(last_oom_error))
            output_meta.extend(meta)

            if not trial_ok:
                case_ok = False
                failure = {
                    "case_index": case_index,
                    "trial": trial,
                    "details": details,
                }
                failed_trials.append(failure)
                case_failures.append(failure)
    except Exception as error:  # pylint: disable=broad-except
        case_ok = False
        failure = {
            "case_index": case_index,
            "trial": current_trial,
            "details": [str(error)],
        }
        failed_trials.append(failure)
        case_failures.append(failure)
        correctness_error_message = str(error)

    generated_profile = _profile_model_case(
        language=language,
        device_id=device_id,
        op_name=op_name,
        case_name=f"verify_generated_case_{case_index}",
        shape=performance_shape,
        model_class=generated_model_class,
        generate_inputs_fn=generate_inputs_fn,
        perf_trials=perf_trials,
    )
    reference_profile = _profile_model_case(
        language=language,
        device_id=device_id,
        op_name=op_name,
        case_name=f"verify_reference_case_{case_index}",
        shape=performance_shape,
        model_class=reference_model_class,
        generate_inputs_fn=generate_inputs_fn,
        perf_trials=perf_trials,
    )
    case_result = {
        "case_index": case_index,
        "device_id": device_id,
        "shape": raw_shape,
        "success": case_ok,
        "failed_trials": case_failures,
        "performance": {
            "generated": generated_profile.get("performance"),
            "reference": reference_profile.get("performance"),
        },
        "profiling": {
            "generated": generated_profile.get("profiling"),
            "reference": reference_profile.get("profiling"),
        },
    }
    if correctness_error_message is not None:
        case_result["error_message"] = correctness_error_message
    case_result["output_meta"] = output_meta[:5]
    performance_errors = {}
    if generated_profile.get("performance_error_message"):
        performance_errors["generated"] = generated_profile["performance_error_message"]
    if reference_profile.get("performance_error_message"):
        performance_errors["reference"] = reference_profile["performance_error_message"]
    if performance_errors:
        case_result["performance_error_message"] = performance_errors

    generated_perf_oom_retries = int(generated_profile.get("oom_retries") or 0)
    reference_perf_oom_retries = int(reference_profile.get("oom_retries") or 0)
    if correctness_oom_retries or generated_perf_oom_retries or reference_perf_oom_retries:
        case_result["retry_meta"] = {
            "correctness_oom_retries": correctness_oom_retries,
            "generated_performance_oom_retries": generated_perf_oom_retries,
            "reference_performance_oom_retries": reference_perf_oom_retries,
        }

    return {
        "case_result": case_result,
        "failed_trials": failed_trials,
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_partial_result(
    *,
    partial_result_path: Optional[Path],
    result: Dict[str, Any],
    status: str,
    partial_result_metadata: Optional[Dict[str, Any]] = None,
    total_cases: Optional[int] = None,
    completed_cases: Optional[int] = None,
    current_case_index: Optional[int] = None,
    last_completed_case_index: Optional[int] = None,
) -> None:
    if partial_result_path is None:
        return

    payload = dict(result)
    if partial_result_metadata:
        payload.update(partial_result_metadata)
    payload["status"] = status
    if total_cases is not None:
        payload["total_cases"] = total_cases
    if completed_cases is not None:
        payload["completed_cases"] = completed_cases
    payload["current_case_index"] = current_case_index
    if last_completed_case_index is not None:
        payload["last_completed_case_index"] = last_completed_case_index
    payload["last_updated"] = datetime.now().isoformat(timespec="seconds")
    ordered_payload = {
        key: payload[key]
        for key in EVALUATION_RESULT_FIELD_ORDER
        if key in payload
    }
    for key, value in payload.items():
        if key not in ordered_payload:
            ordered_payload[key] = value
    _atomic_write_json(partial_result_path, ordered_payload)


def write_partial_generated_operator_result(
    *,
    partial_result_path: Optional[Path],
    result: Dict[str, Any],
    status: str,
    partial_result_metadata: Optional[Dict[str, Any]] = None,
    total_cases: Optional[int] = None,
    completed_cases: Optional[int] = None,
    current_case_index: Optional[int] = None,
    last_completed_case_index: Optional[int] = None,
    running_cases: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write an atomic partial generated-code evaluation result."""
    if partial_result_path is None:
        return
    payload = dict(result)
    if running_cases is not None:
        payload["running_cases"] = running_cases
    _write_partial_result(
        partial_result_path=partial_result_path,
        result=payload,
        status=status,
        partial_result_metadata=partial_result_metadata,
        total_cases=total_cases,
        completed_cases=completed_cases,
        current_case_index=current_case_index,
        last_completed_case_index=last_completed_case_index,
    )


def _base_verification_result(
    *,
    op_name: str,
    language: str,
    shape_mode: str,
    generated_file: Path,
    devices: List[int],
    perf_trials: int,
    trials: int,
    tolerance_policy: Dict[str, Any],
    correctness_policy: Dict[str, Any],
    correctness_baseline_compute_mode: str,
    random_correctness_config: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "op_name": op_name,
        "shape_mode": shape_mode,
        "language": language,
        "devices": list(devices),
        "compiled": None,
        "perf_trials_per_shape": perf_trials,
        "generated_code": str(generated_file),
        "correctness": {
            "success": False,
            "criteria": _correctness_criteria(
                tolerance_policy,
                correctness_policy,
                correctness_baseline_compute_mode,
                random_correctness_config,
                shape_mode,
            ),
            "num_trials": trials,
            "failed_trials": [],
        },
        "cases": [],
    }


def prepare_generated_operator_evaluation(
    *,
    op_name: str,
    language: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    generated_path: str,
    compile_device_id: int,
    devices: List[int],
    trials: int,
    random_seed: Optional[int],
    random_count: int,
    random_max_attempts: int,
    perf_trials: int,
    partial_result_path: Optional[str] = None,
    partial_result_metadata: Optional[Dict[str, Any]] = None,
    torch_extensions_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile-check generated code and resolve shape cases without running cases."""
    operator_spec = get_operator_spec(op_name)
    tolerance_policy = _operator_tolerance_policy(operator_spec)
    correctness_policy = resolve_correctness_policy(operator_spec)
    correctness_baseline_compute_mode = _resolve_correctness_baseline_compute_mode(operator_spec)
    random_correctness_config = _resolve_random_correctness_config(operator_spec)
    generated_file = Path(generated_path)
    partial_path = Path(partial_result_path) if partial_result_path else None
    result = _base_verification_result(
        op_name=op_name,
        language=language,
        shape_mode=shape_mode,
        generated_file=generated_file,
        devices=devices,
        perf_trials=perf_trials,
        trials=trials,
        tolerance_policy=tolerance_policy,
        correctness_policy=correctness_policy,
        correctness_baseline_compute_mode=correctness_baseline_compute_mode,
        random_correctness_config=random_correctness_config,
    )
    write_partial_generated_operator_result(
        partial_result_path=partial_path,
        result=result,
        status="running",
        partial_result_metadata=partial_result_metadata,
        completed_cases=0,
    )

    if not generated_file.exists():
        result["correctness"]["error_message"] = f"Generated code file not found: {generated_file}"
        write_partial_generated_operator_result(
            partial_result_path=partial_path,
            result=result,
            status="failed",
            partial_result_metadata=partial_result_metadata,
            completed_cases=0,
        )
        return {
            "prepared": False,
            "result": result,
            "shapes": [],
            "correctness_baseline_compute_mode": correctness_baseline_compute_mode,
            "random_correctness_config": random_correctness_config,
        }

    compile_result = _run_compile_subprocess(
        op_name=op_name,
        language=language,
        generated_path=str(generated_file),
        device_id=compile_device_id,
        torch_extensions_dir=torch_extensions_dir,
    )
    result["compiled"] = bool(compile_result.get("compiled"))
    write_partial_generated_operator_result(
        partial_result_path=partial_path,
        result=result,
        status="running",
        partial_result_metadata=partial_result_metadata,
        completed_cases=0,
    )
    if not result["compiled"]:
        compile_info = compile_result.get("compile_info") or compile_result.get("error_message")
        result["compile_info"] = compile_info
        result["correctness"]["error_message"] = compile_info
        write_partial_generated_operator_result(
            partial_result_path=partial_path,
            result=result,
            status="failed",
            partial_result_metadata=partial_result_metadata,
            completed_cases=0,
        )
        return {
            "prepared": False,
            "result": result,
            "shapes": [],
            "correctness_baseline_compute_mode": correctness_baseline_compute_mode,
            "random_correctness_config": random_correctness_config,
        }

    if not compile_result.get("has_model_new"):
        result["correctness"]["error_message"] = "Generated code must define ModelNew."
        write_partial_generated_operator_result(
            partial_result_path=partial_path,
            result=result,
            status="failed",
            partial_result_metadata=partial_result_metadata,
            completed_cases=0,
        )
        return {
            "prepared": False,
            "result": result,
            "shapes": [],
            "correctness_baseline_compute_mode": correctness_baseline_compute_mode,
            "random_correctness_config": random_correctness_config,
        }

    validate_shape_fn = load_object(f"{operator_spec['helpers_path']}::validate_shape")
    shapes, random_metadata = _resolve_shapes(
        operator_spec["shape_module"],
        shape_mode,
        shape_name,
        shape_family,
        random_seed,
        random_count,
        random_max_attempts,
        validate_shape_fn,
    )
    if random_metadata is not None:
        result["random"] = random_metadata
    write_partial_generated_operator_result(
        partial_result_path=partial_path,
        result=result,
        status="running",
        partial_result_metadata=partial_result_metadata,
        total_cases=len(shapes),
        completed_cases=0,
    )
    return {
        "prepared": True,
        "result": result,
        "shapes": shapes,
        "correctness_baseline_compute_mode": correctness_baseline_compute_mode,
        "random_correctness_config": random_correctness_config,
    }


def run_generated_operator_case(
    *,
    op_name: str,
    language: str,
    generated_path: str,
    device_id: int,
    case_index: int,
    raw_shape: Dict[str, Any],
    trials: int,
    perf_trials: int,
    correctness_baseline_compute_mode: str,
    torch_extensions_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one generated operator case in an isolated subprocess."""
    return _run_case_subprocess(
        op_name=op_name,
        language=language,
        device_id=device_id,
        generated_path=generated_path,
        case_index=case_index,
        raw_shape=raw_shape,
        trials=trials,
        perf_trials=perf_trials,
        correctness_baseline_compute_mode=correctness_baseline_compute_mode,
        torch_extensions_dir=torch_extensions_dir,
    )


def finalize_generated_operator_result(
    *,
    result: Dict[str, Any],
    shape_mode: str,
    random_correctness_config: Dict[str, float],
    total_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Finalize correctness summary from completed case results."""
    cases = list(result.get("cases") or [])
    cases.sort(key=lambda item: int(item.get("case_index", 0)))
    result["cases"] = cases
    all_ok = True
    failed_trials: List[Dict[str, Any]] = []
    failed_case_count = 0
    for case_result in cases:
        case_ok = bool(case_result.get("success"))
        if not case_ok:
            all_ok = False
            failed_case_count += 1
        failed_trials.extend(case_result.get("failed_trials") or [])

    case_total = len(cases) if total_cases is None else int(total_cases)
    random_case_summary = _random_case_summary(
        shape_mode=shape_mode,
        total_cases=case_total,
        failed_cases=failed_case_count,
        random_correctness_config=random_correctness_config,
    )
    biren_case_compile_failed = (
        str(result.get("language", "")).lower() == "biren"
        and any(
            isinstance(case_result, dict)
            and _is_biren_case_compile_failure(case_result)
            for case_result in cases
        )
    )
    if biren_case_compile_failed:
        message = "Biren case worker reported an extension compilation failure."
        result["compiled"] = False
        result["compile_info"] = message
        result["error_type"] = "CaseCompileFailed"
        result["error_message"] = message
    if shape_mode == "random":
        result["correctness"]["random_case_summary"] = random_case_summary
    result["correctness"]["success"] = (
        random_case_summary["within_tolerance"]
        if shape_mode == "random"
        else all_ok and len(cases) == case_total
    ) and not biren_case_compile_failed
    result["correctness"]["failed_trials"] = failed_trials
    return result


def verify_generated_operator(
    *,
    op_name: str,
    language: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    generated_path: str,
    device_id: int,
    trials: int,
    random_seed: Optional[int],
    random_count: int,
    random_max_attempts: int,
    perf_trials: int,
    partial_result_path: Optional[str] = None,
    partial_result_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify one generated CUDA/SUPA implementation serially on one device."""
    partial_path = Path(partial_result_path) if partial_result_path else None
    try:
        plan = prepare_generated_operator_evaluation(
            op_name=op_name,
            language=language,
            shape_mode=shape_mode,
            shape_name=shape_name,
            shape_family=shape_family,
            generated_path=generated_path,
            compile_device_id=device_id,
            devices=[device_id],
            trials=trials,
            random_seed=random_seed,
            random_count=random_count,
            random_max_attempts=random_max_attempts,
            perf_trials=perf_trials,
            partial_result_path=partial_result_path,
            partial_result_metadata=partial_result_metadata,
        )
        result = plan["result"]
        shapes = plan["shapes"]
        total_cases = len(shapes)
        if not plan.get("prepared"):
            return result

        for case_index, raw_shape in enumerate(shapes):
            write_partial_generated_operator_result(
                partial_result_path=partial_path,
                result=result,
                status="running",
                partial_result_metadata=partial_result_metadata,
                total_cases=total_cases,
                completed_cases=len(result["cases"]),
                current_case_index=case_index,
            )
            case_payload = run_generated_operator_case(
                op_name=op_name,
                language=language,
                device_id=device_id,
                generated_path=generated_path,
                case_index=case_index,
                raw_shape=raw_shape,
                trials=trials,
                perf_trials=perf_trials,
                correctness_baseline_compute_mode=plan["correctness_baseline_compute_mode"],
            )
            case_result = case_payload["case_result"]
            result["cases"].append(case_result)
            result["correctness"]["failed_trials"] = [
                failure
                for case in result["cases"]
                for failure in (case.get("failed_trials") or [])
            ]
            write_partial_generated_operator_result(
                partial_result_path=partial_path,
                result=result,
                status="running",
                partial_result_metadata=partial_result_metadata,
                total_cases=total_cases,
                completed_cases=len(result["cases"]),
                last_completed_case_index=case_index,
            )

        result = finalize_generated_operator_result(
            result=result,
            shape_mode=shape_mode,
            total_cases=total_cases,
            random_correctness_config=plan["random_correctness_config"],
        )
        write_partial_generated_operator_result(
            partial_result_path=partial_path,
            result=result,
            status="done",
            partial_result_metadata=partial_result_metadata,
            total_cases=total_cases,
            completed_cases=len(result["cases"]),
        )
        return result
    except Exception as error:  # pylint: disable=broad-except
        result = locals().get(
            "result",
            {
                "op_name": op_name,
                "shape_mode": shape_mode,
                "language": language,
                "devices": [device_id],
                "compiled": None,
                "perf_trials_per_shape": perf_trials,
                "generated_code": generated_path,
                "correctness": {
                    "success": False,
                    "num_trials": trials,
                    "failed_trials": [],
                },
                "cases": [],
            },
        )
        result["correctness"]["success"] = False
        result["correctness"]["error_message"] = str(error)
        write_partial_generated_operator_result(
            partial_result_path=partial_path,
            result=result,
            status="failed",
            partial_result_metadata=partial_result_metadata,
            completed_cases=len(result["cases"]),
        )
        return result
