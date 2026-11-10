"""Main benchmark orchestration logic."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import signal
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from operators.registry import (
    get_operator_spec,
    load_object,
    load_module,
    resolve_operator_selection,
)
from runner.profiler_runner import (
    ensure_backend,
    profile_case,
)
from runner.random_shape_resolver import (
    generate_random_seed,
    resolve_random_shapes,
)
from runner.random_shape_config import resolve_random_shape_settings
from runner.result_writer import write_result_json
from utils.biren_environment import prepare_biren_environment


ALL_SHAPE_FAMILIES = "all"
CUDA_LIKE_LANGUAGES = {"cuda", "triton", "tiangai", "maca"}


def is_compiled_mode_supported(language: str) -> bool:
    """Return whether torch.compile mode is supported for the backend."""
    return str(language).lower() not in {"biren", "ascendc"}


def resolve_selected_operator_names(args: SimpleNamespace) -> List[str]:
    """Resolve CLI operator selectors into concrete operator names."""
    return resolve_operator_selection(
        op_name=getattr(args, "op", None),
        category=getattr(args, "category", None),
        all_ops=bool(getattr(args, "all", False)),
    )


def is_batch_operator_selection(args: SimpleNamespace) -> bool:
    """Return whether the CLI selection should produce a batch result payload."""
    return bool(
        getattr(args, "category", None)
        or getattr(args, "all", False)
        or len(resolve_selected_operator_names(args)) > 1
    )


def operator_selection_output_name(args: SimpleNamespace) -> str:
    """Return a stable result-file name segment for the selected operators."""
    if getattr(args, "all", False):
        return "all"
    category = getattr(args, "category", None)
    if category:
        return f"category_{category}"
    op_names = resolve_selected_operator_names(args)
    if len(op_names) == 1:
        return op_names[0]
    return f"ops_{len(op_names)}"


def load_torch():
    """Import torch lazily so failures can still produce result JSON."""
    import torch

    return torch

def resolve_shapes(
    shape_module_name: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_seed: Optional[int] = None,
    random_count: int = 100,
    random_max_attempts: int = 10000,
    validate_shape_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Resolve single, family, or random shapes into a unified list."""
    shape_module = load_module(shape_module_name)
    single_shapes = getattr(shape_module, "SINGLE_SHAPES", None)
    shape_families = getattr(shape_module, "SHAPE_FAMILIES", None)

    if single_shapes is None or shape_families is None:
        raise ValueError(
            f"Shape template module {shape_module_name} must define "
            "SINGLE_SHAPES and SHAPE_FAMILIES."
        )

    if shape_mode == "single":
        if not shape_name:
            raise ValueError("shape_name is required when shape_mode is single.")
        if shape_name not in single_shapes:
            raise ValueError(f"Invalid single shape name: {shape_name}")
        return "single_shape", [single_shapes[shape_name]], None

    if shape_mode == "family":
        if not shape_family:
            raise ValueError("shape_family is required when shape_mode is family.")
        if shape_family == ALL_SHAPE_FAMILIES:
            all_shapes: List[Dict[str, int]] = []
            for family_shapes in shape_families.values():
                all_shapes.extend(list(family_shapes))
            return "shape_family", all_shapes, None
        if shape_family not in shape_families:
            raise ValueError(f"Invalid shape family name: {shape_family}")
        return "shape_family", list(shape_families[shape_family]), None

    if shape_mode == "random":
        if validate_shape_fn is None:
            raise ValueError("validate_shape_fn is required when shape_mode is random.")
        resolved_seed = generate_random_seed() if random_seed is None else int(random_seed)
        shapes = resolve_random_shapes(
            shape_module_name,
            seed=resolved_seed,
            count=random_count,
            max_attempts=random_max_attempts,
            validate_shape_fn=validate_shape_fn,
        )
        random_metadata = {
            "seed": resolved_seed,
            "count": random_count,
            "max_attempts": random_max_attempts,
        }
        return "random_shapes", shapes, random_metadata

    raise ValueError(f"Invalid shape_mode: {shape_mode}")


def resolve_language(requested_language: Optional[str]) -> str:
    """Resolve the backend language."""
    language = (requested_language or "cuda").lower()
    ensure_backend(language)
    return language


def resolve_runtime_availability(language: str):
    """Validate runtime availability for the selected backend family."""
    torch = load_torch()
    normalized = str(language).lower()

    if normalized in CUDA_LIKE_LANGUAGES:
        if not torch.cuda.is_available():
            raise ValueError("CUDA-capable GPU was not detected for the selected backend.")
        return

    if normalized == "biren":
        prepare_biren_environment()
        import torch_br  # noqa: F401

        if not torch.supa.is_available():
            raise ValueError("Biren SUPA device was not detected for the selected backend.")
        return

    import torch_npu

    if not hasattr(torch_npu, "npu") or not torch_npu.npu.is_available():
        raise ValueError("NPU device was not detected for the selected backend.")


def resolve_device_ids(language: str, device_ids: List[int]) -> List[int]:
    """Validate requested device ids against the visible devices for the backend."""
    torch = load_torch()
    if not device_ids:
        raise ValueError("device_ids cannot be empty.")

    normalized = str(language).lower()
    if normalized in CUDA_LIKE_LANGUAGES:
        visible_device_count = torch.cuda.device_count()
    elif normalized == "biren":
        prepare_biren_environment()
        import torch_br  # noqa: F401

        visible_device_count = torch.supa.device_count()
    else:
        import torch_npu

        visible_device_count = torch_npu.npu.device_count()

    if visible_device_count <= 0:
        raise ValueError(f"No visible devices were detected for backend {language}.")

    invalid_ids = [
        device_id for device_id in device_ids if device_id < 0 or device_id >= visible_device_count
    ]
    if invalid_ids:
        invalid_text = ", ".join(str(device_id) for device_id in invalid_ids)
        raise ValueError(
            f"Invalid device id(s): {invalid_text}. Visible device count for {language}: {visible_device_count}."
        )
    return device_ids


def get_device_name_by_id(language: str, device_id: int) -> Optional[str]:
    """Return the human-readable device name for a device id."""
    backend = ensure_backend(language)
    previous_context = getattr(backend, "context", {})
    previous_device = getattr(backend, "device", None)
    try:
        if hasattr(backend, "context"):
            backend.context = {}
        if hasattr(backend, "set_device"):
            backend.set_device(device_id)
        device_name = backend.get_hardware_name() if hasattr(backend, "get_hardware_name") else None
        return device_name or None
    finally:
        if hasattr(backend, "context"):
            backend.context = previous_context
        if previous_device is not None:
            backend.device = previous_device


def format_device_names(language: str, device_ids: List[int]) -> Optional[str]:
    """Format the allowed devices into a single result string."""
    device_names = []
    for device_id in device_ids:
        device_name = get_device_name_by_id(language, device_id)
        if device_name is not None:
            device_names.append(f"{device_name} ({language}:{device_id})")
    if not device_names:
        return None
    return ", ".join(device_names)


def build_case_result(
    shape: Optional[Dict[str, Any]],
    success: bool,
    error_message: str,
    performance: Optional[Dict[str, Any]],
    profiling: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a standardized per-case result structure."""
    return {
        "shape": public_shape(shape),
        "success": success,
        "error_message": error_message,
        "performance": performance,
        "profiling": profiling,
    }


def public_shape(shape: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove framework-internal shape fields from result JSON."""
    if shape is None:
        return None
    return {key: value for key, value in shape.items() if not key.startswith("_")}


def run_case(
    language: str,
    op_name: str,
    case_name: str,
    device_id: int,
    model_class: type,
    generate_inputs_fn: Callable[..., Tuple[Any, ...]],
    shape: Dict[str, Any],
    perf_trials: int,
    use_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Run a single benchmark case through backend timing."""
    profiler_summary = profile_case(
        language=language,
        device_id=device_id,
        op_name=op_name,
        case_name=case_name,
        shape=shape,
        model_class=model_class,
        generate_inputs_fn=generate_inputs_fn,
        perf_trials=perf_trials,
        use_torch_compile=use_torch_compile,
    )
    return build_case_result(
        shape=shape,
        success=True,
        error_message="",
        performance=profiler_summary.get("performance"),
        profiling=profiler_summary.get("profiling"),
    )


def create_failure_result(
    op_name: str,
    shape_mode: str,
    language: Optional[str],
    device_name: Optional[str],
    error_message: str,
    shape: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a top-level result payload for benchmark failures."""
    return {
        "op_name": op_name,
        "shape_mode": shape_mode,
        "language": language,
        "device": device_name,
        "cases": [
            build_case_result(
                shape=shape,
                success=False,
                error_message=error_message,
                performance=None,
                profiling=None,
            )
        ],
    }


def shape_with_baseline_compute_mode(
    shape: Dict[str, Any],
    baseline_compute_mode: str,
) -> Dict[str, Any]:
    """Copy a shape and inject the reference baseline compute mode."""
    case_shape = dict(shape)
    case_shape["_baseline_compute_mode"] = baseline_compute_mode
    return case_shape


def _run_single_operator_benchmark_worker(
    result_queue: Any,
    op_name: str,
    language: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_seed: Optional[int],
    random_count: Optional[int],
    random_max_attempts: Optional[int],
    perf_trials: int,
    baseline_compute_mode: str,
    workers: int,
    allowed_device_ids: List[int],
    resolved_device_name: Optional[str],
    use_torch_compile: bool,
) -> None:
    """Execute one operator benchmark in an isolated subprocess."""
    result = run_single_operator_benchmark(
        op_name=op_name,
        language=language,
        shape_mode=shape_mode,
        shape_name=shape_name,
        shape_family=shape_family,
        random_seed=random_seed,
        random_count=random_count,
        random_max_attempts=random_max_attempts,
        perf_trials=perf_trials,
        baseline_compute_mode=baseline_compute_mode,
        workers=workers,
        allowed_device_ids=allowed_device_ids,
        resolved_device_name=resolved_device_name,
        use_torch_compile=use_torch_compile,
    )
    result_queue.put(result)


def _format_process_exit_error(op_name: str, exitcode: int) -> str:
    """Format an informative error when an isolated operator subprocess crashes."""
    if exitcode < 0:
        signal_num = -exitcode
        try:
            signal_name = signal.Signals(signal_num).name
        except ValueError:
            signal_name = f"SIG{signal_num}"
        return (
            f"Operator {op_name} crashed in an isolated subprocess with signal "
            f"{signal_name} ({signal_num})."
        )
    return f"Operator {op_name} crashed in an isolated subprocess with exit code {exitcode}."


def run_single_operator_benchmark_isolated(
    op_name: str,
    language: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_seed: Optional[int],
    random_count: Optional[int],
    random_max_attempts: Optional[int],
    perf_trials: int,
    baseline_compute_mode: str,
    workers: int,
    allowed_device_ids: List[int],
    resolved_device_name: Optional[str],
    use_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Run one operator benchmark in a subprocess so fatal backend crashes are contained."""
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    process = spawn_context.Process(
        target=_run_single_operator_benchmark_worker,
        args=(
            result_queue,
            op_name,
            language,
            shape_mode,
            shape_name,
            shape_family,
            random_seed,
            random_count,
            random_max_attempts,
            perf_trials,
            baseline_compute_mode,
            workers,
            allowed_device_ids,
            resolved_device_name,
            use_torch_compile,
        ),
    )
    process.start()
    process.join()

    result: Optional[Dict[str, Any]] = None
    if process.exitcode == 0:
        try:
            if not result_queue.empty():
                result = result_queue.get_nowait()
        except Exception:  # pylint: disable=broad-except
            result = None

    result_queue.close()
    result_queue.join_thread()

    if result is not None:
        return result

    return create_failure_result(
        op_name=op_name,
        shape_mode=shape_mode,
        language=language,
        device_name=resolved_device_name,
        error_message=_format_process_exit_error(op_name, process.exitcode or 0),
    )


def _run_operator_batch_on_device(
    batch: List[Tuple[int, str]],
    *,
    device_id: int,
    language: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_seed: Optional[int],
    random_count: Optional[int],
    random_max_attempts: Optional[int],
    perf_trials: int,
    baseline_compute_mode: str,
    use_torch_compile: bool,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Run a sequence of operators on a single dedicated device."""
    results: List[Tuple[int, Dict[str, Any]]] = []
    device_name = format_device_names(language, [device_id])
    for result_index, op_name in batch:
        result = run_single_operator_benchmark_isolated(
            op_name=op_name,
            language=language,
            shape_mode=shape_mode,
            shape_name=shape_name,
            shape_family=shape_family,
            random_seed=random_seed,
            random_count=random_count,
            random_max_attempts=random_max_attempts,
            perf_trials=perf_trials,
            baseline_compute_mode=baseline_compute_mode,
            workers=1,
            allowed_device_ids=[device_id],
            resolved_device_name=device_name,
            use_torch_compile=use_torch_compile,
        )
        results.append((result_index, result))
    return results


def run_selected_operators_benchmark_parallel(
    args: SimpleNamespace,
    *,
    operator_names: List[str],
    language: str,
    allowed_device_ids: List[int],
    resolved_device_name: Optional[str],
    use_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Run selected operators by distributing operators across visible devices."""
    operator_results: List[Optional[Dict[str, Any]]] = [None] * len(operator_names)
    device_batches: Dict[int, List[Tuple[int, str]]] = {device_id: [] for device_id in allowed_device_ids}

    for result_index, op_name in enumerate(operator_names):
        device_id = allowed_device_ids[result_index % len(allowed_device_ids)]
        device_batches[device_id].append((result_index, op_name))

    with ThreadPoolExecutor(max_workers=len(allowed_device_ids)) as executor:
        futures = [
            executor.submit(
                _run_operator_batch_on_device,
                batch,
                device_id=device_id,
                language=language,
                shape_mode=args.shape_mode,
                shape_name=args.shape_name,
                shape_family=args.shape_family,
                random_seed=getattr(args, "random_seed", None),
                random_count=getattr(args, "random_count", None),
                random_max_attempts=getattr(args, "random_max_attempts", None),
                perf_trials=getattr(args, "perf_trials", 5),
                baseline_compute_mode=getattr(args, "baseline_compute_mode", "performance"),
                use_torch_compile=use_torch_compile,
            )
            for device_id, batch in device_batches.items()
            if batch
        ]

        for future in futures:
            for result_index, result in future.result():
                operator_results[result_index] = result

    finalized_results = [
        result
        if result is not None
        else create_failure_result(
            op_name=operator_names[result_index],
            shape_mode=args.shape_mode,
            language=language,
            device_name=resolved_device_name,
            error_message="Operator did not produce a result.",
        )
        for result_index, result in enumerate(operator_results)
    ]
    return {"ops": finalized_results}


def build_case_task(
    op_name: str,
    language: str,
    shape: Dict[str, Any],
    device_id: int,
    case_index: int,
    perf_trials: int,
    use_torch_compile: bool,
) -> Dict[str, Any]:
    """Build a serializable worker task for a single shape case."""
    operator_spec = get_operator_spec(op_name)
    return {
        "case_index": case_index,
        "shape": shape,
        "language": language,
        "op_name": op_name,
        "device_id": device_id,
        "perf_trials": perf_trials,
        "use_torch_compile": use_torch_compile,
        "model_class_path": operator_spec["model_class_path"],
        "generate_inputs_path": operator_spec["generate_inputs_path"],
    }


def execute_case_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single case task inside a worker process."""
    model_class = load_object(task["model_class_path"])
    generate_inputs_fn = load_object(task["generate_inputs_path"])
    case_name = f"case_{task['case_index']}"

    try:
        case_result = run_case(
            language=task["language"],
            op_name=task["op_name"],
            case_name=case_name,
            device_id=task["device_id"],
            model_class=model_class,
            generate_inputs_fn=generate_inputs_fn,
            shape=task["shape"],
            perf_trials=task["perf_trials"],
            use_torch_compile=task["use_torch_compile"],
        )
    except Exception as case_error:  # pylint: disable=broad-except
        case_result = build_case_result(
            shape=task["shape"],
            success=False,
            error_message=str(case_error),
            performance=None,
            profiling=None,
        )

    return {
        "case_index": task["case_index"],
        "case_result": case_result,
    }


def _initialize_biren_case_worker(device_id: int) -> None:
    """Pin one fresh worker process to a single physical Biren device."""
    prepare_biren_environment(device_id)


def _run_biren_device_case_tasks(
    device_id: int,
    case_tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run Biren cases in a reusable process that never changes physical device."""
    spawn_context = mp.get_context("spawn")
    with spawn_context.Pool(
        processes=1,
        initializer=_initialize_biren_case_worker,
        initargs=(device_id,),
    ) as pool:
        return pool.map(execute_case_task, case_tasks)


def run_case_tasks(case_tasks: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """Run case tasks sequentially or with a worker-process pool."""
    if not case_tasks:
        return []

    if str(case_tasks[0].get("language", "")).lower() == "biren":
        tasks_by_device: Dict[int, List[Dict[str, Any]]] = {}
        for task in case_tasks:
            device_id = int(task["device_id"])
            tasks_by_device.setdefault(device_id, []).append(task)

        max_device_workers = max(1, min(int(workers), len(tasks_by_device)))
        completed: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_device_workers) as executor:
            futures = [
                executor.submit(_run_biren_device_case_tasks, device_id, device_tasks)
                for device_id, device_tasks in tasks_by_device.items()
            ]
            for future in futures:
                completed.extend(future.result())
        return completed

    if workers == 1:
        return [execute_case_task(task) for task in case_tasks]

    spawn_context = mp.get_context("spawn")
    with spawn_context.Pool(processes=workers) as pool:
        return pool.map(execute_case_task, case_tasks)


def build_aggregate_performance(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-shape timing into a random/family aggregate."""
    total_cases = len(cases)
    successful_cases = 0
    failed_cases = 0
    total_mean_ms = 0.0
    missing_performance_cases = 0

    for case in cases:
        if not case.get("success"):
            failed_cases += 1
            continue
        successful_cases += 1
        performance = case.get("performance") or {}
        mean_ms = performance.get("mean")
        if mean_ms is None:
            missing_performance_cases += 1
            continue
        total_mean_ms += float(mean_ms)

    return {
        "total_mean_ms": round(total_mean_ms, 6),
        "total_cases": total_cases,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "missing_performance_cases": missing_performance_cases,
        "complete": failed_cases == 0 and missing_performance_cases == 0,
    }


def run_single_operator_benchmark(
    op_name: str,
    language: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_seed: Optional[int],
    random_count: Optional[int],
    random_max_attempts: Optional[int],
    perf_trials: int,
    baseline_compute_mode: str,
    workers: int,
    allowed_device_ids: List[int],
    resolved_device_name: Optional[str],
    use_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Run benchmark flow for a single registered operator without writing files."""
    try:
        operator_spec = get_operator_spec(op_name)
        resolved_random_count, resolved_random_max_attempts = resolve_random_shape_settings(
            operator_spec,
            random_count=random_count,
            random_max_attempts=random_max_attempts,
        )
        validate_shape_fn = load_object(f"{operator_spec['helpers_path']}::validate_shape")
        _, shapes, random_metadata = resolve_shapes(
            operator_spec["shape_module"],
            shape_mode,
            shape_name,
            shape_family,
            random_seed=random_seed,
            random_count=resolved_random_count,
            random_max_attempts=resolved_random_max_attempts,
            validate_shape_fn=validate_shape_fn,
        )

        case_tasks = [
            build_case_task(
                op_name=op_name,
                language=language,
                shape=shape_with_baseline_compute_mode(shape, baseline_compute_mode),
                device_id=allowed_device_ids[index % len(allowed_device_ids)],
                case_index=index,
                perf_trials=perf_trials,
                use_torch_compile=use_torch_compile,
            )
            for index, shape in enumerate(shapes)
        ]

        completed_cases = run_case_tasks(case_tasks, workers=workers)
        completed_cases.sort(key=lambda item: item["case_index"])

        cases = [item["case_result"] for item in completed_cases]
        result = {
            "op_name": op_name,
            "shape_mode": shape_mode,
            "language": language,
            "device": resolved_device_name,
            "baseline_compute_mode": baseline_compute_mode,
            "perf_trials_per_shape": perf_trials,
            "cases": cases,
        }
        if random_metadata is not None:
            random_metadata["perf_trials_per_shape"] = perf_trials
            result["random"] = random_metadata
        if shape_mode == "random":
            result["aggregate_performance"] = build_aggregate_performance(cases)
        return result

    except Exception as error:  # pylint: disable=broad-except
        return create_failure_result(
            op_name=op_name,
            shape_mode=shape_mode,
            language=language,
            device_name=resolved_device_name,
            error_message=str(error),
        )


def run_all_operators_benchmark(
    args: SimpleNamespace,
    *,
    use_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Run selected registered operators using the requested shape mode and merge results."""
    operator_names = resolve_selected_operator_names(args)
    resolved_device_name: Optional[str] = None
    try:
        language = resolve_language(args.language)
        resolve_runtime_availability(language)
        allowed_device_ids = resolve_device_ids(language, args.device_ids)
        resolved_device_name = format_device_names(language, allowed_device_ids)
        return run_selected_operators_benchmark_parallel(
            args,
            operator_names=operator_names,
            language=language,
            allowed_device_ids=allowed_device_ids,
            resolved_device_name=resolved_device_name,
            use_torch_compile=use_torch_compile,
        )

    except Exception as error:  # pylint: disable=broad-except
        return {
            "ops": [
                create_failure_result(
                    op_name=op_name,
                    shape_mode=args.shape_mode,
                    language=getattr(args, "language", None),
                    device_name=resolved_device_name,
                    error_message=str(error),
                )
                for op_name in operator_names
            ],
        }


def _run_benchmark_once(
    args: SimpleNamespace,
    *,
    execution_mode: str,
    use_torch_compile: bool,
) -> Dict[str, Any]:
    """Run one benchmark pass for a single execution mode."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_language = resolve_language(getattr(args, "language", None))
    operator_names = resolve_selected_operator_names(args)
    batch_selection = is_batch_operator_selection(args)

    if use_torch_compile and not is_compiled_mode_supported(resolved_language):
        result_payload = (
            {
                "ops": [
                    create_failure_result(
                        op_name=op_name,
                        shape_mode=args.shape_mode,
                        language=resolved_language,
                        device_name=None,
                        error_message="Compiled mode is not supported for backend biren. Use --mode eager.",
                    )
                    for op_name in operator_names
                ]
            }
            if batch_selection
            else create_failure_result(
                op_name=operator_names[0],
                shape_mode=args.shape_mode,
                language=resolved_language,
                device_name=None,
                error_message="Compiled mode is not supported for backend biren. Use --mode eager.",
            )
        )
        output_op_name = operator_selection_output_name(args) if batch_selection else result_payload["op_name"]
        output_mode = args.shape_mode if batch_selection else result_payload["shape_mode"]
        output_path = write_result_json(
            base_dir=base_dir,
            op_name=output_op_name,
            mode=output_mode,
            result=result_payload,
            execution_mode=execution_mode,
        )
        result_payload["result_file"] = output_path
        return result_payload

    if batch_selection:
        result_payload = run_all_operators_benchmark(args, use_torch_compile=use_torch_compile)
        output_op_name = operator_selection_output_name(args)
        output_mode = args.shape_mode
    else:
        resolved_device_name: Optional[str] = None
        op_name = operator_names[0]
        try:
            language = resolve_language(args.language)
            if use_torch_compile and not is_compiled_mode_supported(language):
                raise ValueError(
                    "Compiled mode is not supported for backend biren. Use --mode eager."
                )
            resolve_runtime_availability(language)
            allowed_device_ids = resolve_device_ids(language, args.device_ids)
            resolved_device_name = format_device_names(language, allowed_device_ids)
            result_payload = run_single_operator_benchmark(
                op_name=op_name,
                language=language,
                shape_mode=args.shape_mode,
                shape_name=args.shape_name,
                shape_family=args.shape_family,
                random_seed=getattr(args, "random_seed", None),
                random_count=getattr(args, "random_count", None),
                random_max_attempts=getattr(args, "random_max_attempts", None),
                perf_trials=getattr(args, "perf_trials", 5),
                baseline_compute_mode=getattr(args, "baseline_compute_mode", "performance"),
                workers=args.workers,
                allowed_device_ids=allowed_device_ids,
                resolved_device_name=resolved_device_name,
                use_torch_compile=use_torch_compile,
            )
        except Exception as error:  # pylint: disable=broad-except
            result_payload = create_failure_result(
                op_name=op_name,
                shape_mode=args.shape_mode,
                language=getattr(args, "language", None),
                device_name=resolved_device_name,
                error_message=str(error),
            )
        output_op_name = result_payload["op_name"]
        output_mode = result_payload["shape_mode"]

    output_path = write_result_json(
        base_dir=base_dir,
        op_name=output_op_name,
        mode=output_mode,
        result=result_payload,
        execution_mode=execution_mode,
    )
    result_payload["result_file"] = output_path
    return result_payload


def run_benchmark(args: SimpleNamespace) -> Dict[str, Any]:
    """Run a benchmark end-to-end and emit one or two result JSON files."""
    if args.mode not in {"eager", "compiled", "both"}:
        raise ValueError("run_benchmark only supports eager, compiled, or both modes.")

    normalized_language = resolve_language(getattr(args, "language", None))

    if args.mode == "compiled" and not is_compiled_mode_supported(normalized_language):
        return {
            "runs": [
                {
                    "execution_mode": "compiled",
                    "result": _run_benchmark_once(
                        args,
                        execution_mode="compiled",
                        use_torch_compile=True,
                    ),
                }
            ]
        }

    if args.mode == "both":
        eager_result = _run_benchmark_once(
            args,
            execution_mode="eager",
            use_torch_compile=False,
        )
        compiled_result = _run_benchmark_once(
            args,
            execution_mode="compiled",
            use_torch_compile=True,
        )
        return {
            "runs": [
                {"execution_mode": "eager", "result": eager_result},
                {"execution_mode": "compiled", "result": compiled_result},
            ]
        }

    use_torch_compile = args.mode == "compiled"
    return _run_benchmark_once(
        args,
        execution_mode=args.mode,
        use_torch_compile=use_torch_compile,
    )
