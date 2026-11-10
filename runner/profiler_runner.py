"""Benchmark profiling helpers built on top of local backends."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional

from backends.backend_registry import BACKEND_REGISTRY
from utils.biren_environment import prepare_biren_environment


def ensure_backend(language: str):
    """Load and return a local backend instance by language name."""
    normalized = str(language).lower()
    if normalized not in BACKEND_REGISTRY:
        try:
            importlib.import_module(f"backends.{normalized}_backend")
        except ImportError as error:
            raise ValueError(f"Unsupported language/platform: {language}") from error

    backend = BACKEND_REGISTRY.get(normalized)
    if backend is None:
        raise ValueError(f"Unsupported language/platform: {language}")
    return backend


def _load_runtime_module(language: str):
    """Import and return the runtime module associated with a backend."""
    normalized = str(language).lower()
    if normalized in {"cuda", "triton", "tiangai", "maca"}:
        import torch

        return torch
    if normalized == "biren":
        prepare_biren_environment()
        import torch
        import torch_br  # noqa: F401

        return torch

    import torch
    import torch_npu  # noqa: F401

    return torch


def setup_backend_device(backend, language: str, device_id: int) -> Any:
    """Configure the backend onto a concrete device and return that device object."""
    if str(language).lower() == "biren":
        prepare_biren_environment(device_id)
    torch = _load_runtime_module(language)
    normalized = str(language).lower()

    if hasattr(backend, "setup_device"):
        return backend.setup_device(device_id)

    if hasattr(backend, "set_device"):
        backend.set_device(device_id)
        device = backend.get_device() if hasattr(backend, "get_device") else None
        if device is not None:
            return device

    if normalized in {"cuda", "triton", "tiangai", "maca"}:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        backend.device = device
        return device

    if normalized == "biren":
        torch.supa.set_device(device_id)
        device = torch.device(f"supa:{device_id}")
        backend.device = device
        return device

    import torch_npu

    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    backend.device = device
    return device


def profile_case(
    *,
    language: str,
    device_id: int,
    op_name: str,
    case_name: str,
    shape: Dict[str, Any],
    model_class: type,
    generate_inputs_fn: Callable[..., tuple[Any, ...]],
    perf_trials: Optional[int] = None,
    use_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Run a single benchmark case through backend timing."""
    backend = ensure_backend(language)
    backend.context = {}
    device = setup_backend_device(backend, language, device_id)
    normalized = str(language).lower()

    def get_inputs():
        inputs = generate_inputs_fn(shape)
        if isinstance(inputs, tuple):
            return list(inputs)
        if isinstance(inputs, list):
            return inputs
        return [inputs]

    def get_init_inputs():
        return [shape]

    backend.context["Model"] = model_class
    backend.context["get_inputs"] = get_inputs
    backend.context["get_init_inputs"] = get_init_inputs
    backend.context["_op_eval_op_name"] = f"op_benchmark/{op_name}/{case_name}/operator_defined"

    previous_trials = os.environ.get("OP_EVAL_CUDA_NUM_TRIALS")
    if perf_trials is not None:
        os.environ["OP_EVAL_CUDA_NUM_TRIALS"] = str(perf_trials)

    try:
        if use_torch_compile:
            return _profile_case_compiled(
                language=normalized,
                device=device,
                op_tag=backend.context["_op_eval_op_name"],
                model_class=model_class,
                get_inputs=get_inputs,
                get_init_inputs=get_init_inputs,
                perf_trials=perf_trials,
            )
        _, profiling_result = backend.time_execution(eval_target="Model")
        if profiling_result is None:
            profiling_result = {"performance": None, "profiling": None}
        return profiling_result
    finally:
        if previous_trials is None:
            os.environ.pop("OP_EVAL_CUDA_NUM_TRIALS", None)
        else:
            os.environ["OP_EVAL_CUDA_NUM_TRIALS"] = previous_trials
        if hasattr(backend, "cleanup"):
            backend.cleanup()


def _profile_case_compiled(
    *,
    language: str,
    device,
    op_tag: str,
    model_class: type,
    get_inputs: Callable[[], list[Any]],
    get_init_inputs: Callable[[], list[Any]],
    perf_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single benchmark case via torch.compile before timing."""
    import torch

    from runner.config import NUM_PERF_TRIALS, NUM_WARMUP
    from utils.performance import (
        _build_cuda_perf_result,
        _time_execution_cuda_event,
        profile_execution,
        set_perf_seed,
    )

    inputs = get_inputs()
    init_inputs = get_init_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    set_perf_seed(op_tag)
    with torch.no_grad():
        model = model_class(*init_inputs).to(device)
        model.eval()
        torch._dynamo.reset()
        model = torch.compile(model, backend="inductor", mode="default")

        if language in {"cuda", "triton", "tiangai", "maca"}:
            previous_warmup = os.environ.get("OP_EVAL_CUDA_NUM_WARMUP")
            os.environ["OP_EVAL_CUDA_NUM_WARMUP"] = str(NUM_WARMUP)
            try:
                elapsed_times_ms = _time_execution_cuda_event(
                    lambda: model(*inputs),
                    device=device,
                )
            finally:
                if previous_warmup is None:
                    os.environ.pop("OP_EVAL_CUDA_NUM_WARMUP", None)
                else:
                    os.environ["OP_EVAL_CUDA_NUM_WARMUP"] = previous_warmup
            return _build_cuda_perf_result(elapsed_times_ms)

        if language == "biren":
            warmup = int(os.environ.get("OP_EVAL_CUDA_NUM_WARMUP", str(NUM_WARMUP)))
            trials = int(os.environ.get("OP_EVAL_CUDA_NUM_TRIALS", str(perf_trials or NUM_PERF_TRIALS)))
            discard_first = int(os.environ.get("OP_EVAL_CUDA_DISCARD_FIRST", "1"))
            warmup = max(0, warmup)
            trials = max(1, trials)
            discard_first = max(0, discard_first)

            elapsed_times_ms = []
            for _ in range(warmup):
                model(*inputs)
                torch.supa.synchronize(device=device)

            torch.supa.empty_cache()

            for trial_idx in range(trials + discard_first):
                torch.supa.synchronize(device=device)
                start = torch.supa.Event(enable_timing=True)
                end = torch.supa.Event(enable_timing=True)
                start.record()
                model(*inputs)
                end.record()
                torch.supa.synchronize(device=device)
                if trial_idx >= discard_first:
                    elapsed_times_ms.append(float(start.elapsed_time(end)))

            perf_result = None
            if elapsed_times_ms:
                import numpy as np

                perf_result = {
                    "mean": round(float(np.mean(elapsed_times_ms)), 3),
                    "std": round(float(np.std(elapsed_times_ms)), 3),
                    "min": round(float(np.min(elapsed_times_ms)), 3),
                    "max": round(float(np.max(elapsed_times_ms)), 3),
                    "num_trials": len(elapsed_times_ms),
                }

            return {
                "performance": perf_result,
                "profiling": None,
            }

        import torch_npu

        for _ in range(NUM_WARMUP):
            model(*inputs)
            torch_npu.npu.synchronize(device=device)

        profiling_result = profile_execution(lambda: model(*inputs), device, op_tag)
        if profiling_result is None:
            profiling_result = {"performance": None, "profiling": None}
        return profiling_result
