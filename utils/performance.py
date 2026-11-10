"""CUDA timing helpers used by op_benchmark."""

import hashlib
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from runner.config import NUM_PERF_TRIALS, NUM_WARMUP, SEED_NUM


_PREHEATED_CUDA_DEVICES = set()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _round3(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 3)


def _normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    raise TypeError(f"Unsupported device type: {type(device)}")


def perf_seed_for_op(op_name: str) -> int:
    seed_salt = os.environ.get("OP_EVAL_PERF_SEED_SALT", "op_benchmark_perf_seed")
    seed_base = int(os.environ.get("OP_EVAL_PERF_SEED_BASE", str(SEED_NUM)))
    digest = hashlib.sha256(f"{seed_salt}:{op_name}".encode("utf-8")).digest()
    hashed = int.from_bytes(digest[:8], "little", signed=False)
    return (seed_base + hashed) % (2**31 - 1)


def set_perf_seed(op_name: Optional[str]) -> None:
    """Set deterministic seeds for performance inputs."""
    if not op_name:
        return
    seed = perf_seed_for_op(op_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def profile_execution(
    run_fn: Callable,
    device,
    op_name: str,
    output_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Placeholder for profiler-style execution.

    CUDA timing in this benchmark uses CUDA events directly, so this function is
    kept only for API compatibility with backend timing helpers.
    """
    del run_fn, device, op_name, output_root
    return None


def _clear_l2_cache_cuda(device: torch.device) -> None:
    """Best-effort CUDA L2 cache thrash."""
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def _preheat_cuda_device(device: torch.device) -> None:
    """Run a fixed large FP16 matmul once per process/device before timing."""
    device = _normalize_device(device)
    with torch.cuda.device(device):
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if device_index in _PREHEATED_CUDA_DEVICES:
            return

        a = torch.randn((8192, 8192), dtype=torch.float16, device=device)
        b = torch.randn((8192, 8192), dtype=torch.float16, device=device)
        c = torch.empty((8192, 8192), dtype=torch.float16, device=device)
        torch.mm(a, b, out=c)
        torch.cuda.synchronize(device=device)
        del a, b, c

        _PREHEATED_CUDA_DEVICES.add(device_index)


def _time_execution_cuda_event(
    run_once: Callable[[], None],
    *,
    device: torch.device,
) -> List[float]:
    """Measure a callable with CUDA event timing."""
    warmup = int(os.environ.get("OP_EVAL_CUDA_NUM_WARMUP", str(NUM_WARMUP)))
    trials = int(os.environ.get("OP_EVAL_CUDA_NUM_TRIALS", str(NUM_PERF_TRIALS)))
    discard_first = int(os.environ.get("OP_EVAL_CUDA_DISCARD_FIRST", "1"))
    use_l2_flush = _bool_env("OP_EVAL_CUDA_CLEAR_L2_CACHE", True)

    warmup = max(0, warmup)
    trials = max(1, trials)
    discard_first = max(0, discard_first)

    elapsed_times_ms: List[float] = []
    l2_flush_available = use_l2_flush

    with torch.cuda.device(device):
        _preheat_cuda_device(device)

        for _ in range(warmup):
            run_once()
            torch.cuda.synchronize(device=device)

        torch.cuda.empty_cache()

        for trial_idx in range(trials + discard_first):
            torch.cuda.synchronize(device=device)

            if l2_flush_available:
                try:
                    _clear_l2_cache_cuda(device)
                except Exception:
                    l2_flush_available = False

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            run_once()
            end_event.record()
            torch.cuda.synchronize(device=device)

            if trial_idx >= discard_first:
                elapsed_times_ms.append(float(start_event.elapsed_time(end_event)))

    return elapsed_times_ms


def _build_cuda_perf_result(elapsed_times_ms: List[float]) -> Dict[str, Any]:
    """Build the standard performance result payload from CUDA samples."""
    if not elapsed_times_ms:
        return {"performance": None, "profiling": None}
    performance = {
        "mean": _round3(float(np.mean(elapsed_times_ms))),
        "std": _round3(float(np.std(elapsed_times_ms))),
        "min": _round3(float(np.min(elapsed_times_ms))),
        "max": _round3(float(np.max(elapsed_times_ms))),
        "num_trials": len(elapsed_times_ms),
    }
    return {"performance": performance, "profiling": None}


def _time_execution_cuda_event_template(
    context,
    device,
    eval_target,
) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Build and time the model with CUDA event timing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested but torch.cuda.is_available() is False.")

    device = _normalize_device(device)
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device}.")

    get_inputs = context["get_inputs"]
    get_init_inputs = context["get_init_inputs"]
    model_class = context[eval_target]

    set_perf_seed(context.get("_op_eval_op_name"))

    inputs = get_inputs()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    init_inputs = get_init_inputs()
    init_inputs = [x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    with torch.no_grad():
        model = model_class(*init_inputs).to(device).eval()

        def run_once():
            model(*inputs)

        elapsed_times_ms = _time_execution_cuda_event(run_once, device=device)

    return elapsed_times_ms, _build_cuda_perf_result(elapsed_times_ms)


def time_execution_event_template(
    context,
    device,
    synchronize,
    event_class,
    eval_target,
) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Backend-compatible timing entrypoint."""
    del synchronize, event_class
    device_obj = _normalize_device(device)
    if device_obj.type != "cuda":
        raise ValueError(f"Only CUDA timing is supported by standalone op_benchmark, got {device_obj}.")
    return _time_execution_cuda_event_template(
        context=context,
        device=device_obj,
        eval_target=eval_target,
    )
