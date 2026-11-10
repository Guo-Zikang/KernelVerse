"""Retry failed generation records from an existing run directory."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

from codegen.artifacts import write_json
from codegen.batch_runner import PASSK_TEMPERATURE
from codegen.generators import GenerationRequest, run_generation_with_retries
from operators.registry import get_operator_spec
from utils.run_paths import existing_op_run_dir


LogFn = Callable[..., None]


@dataclass(frozen=True)
class FailedGenerationTask:
    """One failed leaf generation record that can be retried."""

    generation_path: Path
    op_name: str
    run_id: str
    run_dir: Path
    shape_mode: str
    shape_name: str
    shape_family: str
    language: str
    device_id: int
    random_count: int | None
    random_max_attempts: int | None
    random_seed: int | None
    candidate_id: int | None
    samples_per_op: int


@dataclass(frozen=True)
class FailedGenerationRetryRequest:
    """Configuration for retrying failed generations in an existing run."""

    run_dir: Path
    llm_workers: int = 200
    generation_retries: int = 5
    env_path: Path | None = None
    device_ids: list[int] | None = None


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _candidate_id_from_dir(path: Path) -> int | None:
    prefix = "candidate_"
    if not path.name.startswith(prefix):
        return None
    try:
        return int(path.name[len(prefix):])
    except ValueError:
        return None


def _task_from_generation(
    *,
    generation_path: Path,
    generation: dict,
    batch: dict | None,
    op_record: dict | None,
    candidate_record: dict | None,
    device_id: int,
) -> FailedGenerationTask:
    op_name = generation.get("op_name") or (op_record or {}).get("op_name")
    if not isinstance(op_name, str) or not op_name:
        raise ValueError(f"Missing op_name in failed generation: {generation_path}")

    shape_mode = generation.get("shape_mode") or (batch or {}).get("shape_mode")
    if shape_mode not in {"single", "family", "random"}:
        raise ValueError(
            f"Missing or invalid shape_mode in failed generation: {generation_path}"
        )

    candidate_id = generation.get("candidate_id")
    if candidate_id is None:
        candidate_id = (candidate_record or {}).get("candidate_id")
    if candidate_id is None and generation_path.parent.parent.name == "candidates":
        candidate_id = _candidate_id_from_dir(generation_path.parent)
    candidate_id = int(candidate_id) if candidate_id is not None else None

    samples_per_op = int(
        generation.get("samples_per_op")
        or (op_record or {}).get("samples_per_op")
        or (batch or {}).get("samples_per_op")
        or 1
    )
    random_shape = generation.get("random_shape") or {}
    if not isinstance(random_shape, dict):
        random_shape = {}

    return FailedGenerationTask(
        generation_path=generation_path,
        op_name=op_name,
        run_id=str(generation.get("run_id") or f"{generation_path.parent.name}_{op_name}"),
        run_dir=generation_path.parent,
        shape_mode=shape_mode,
        shape_name=str(
            generation.get("shape_name")
            or (batch or {}).get("shape_name")
            or "default"
        ),
        shape_family=str(
            generation.get("shape_family")
            or (batch or {}).get("shape_family")
            or "all"
        ),
        language=str(generation.get("language") or (batch or {}).get("language") or "cuda"),
        device_id=device_id,
        random_count=random_shape.get("count"),
        random_max_attempts=random_shape.get("max_attempts"),
        random_seed=random_shape.get("seed", 42),
        candidate_id=candidate_id,
        samples_per_op=samples_per_op,
    )


def _candidate_records_by_id(op_record: dict) -> dict[int, dict]:
    records = {}
    for record in op_record.get("candidates") or []:
        if isinstance(record, dict) and record.get("candidate_id") is not None:
            records[int(record["candidate_id"])] = record
    return records


def discover_failed_generation_tasks(
    request: FailedGenerationRetryRequest,
) -> list[FailedGenerationTask]:
    """Find failed leaf generation records under a single or batch run."""
    run_dir = request.run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    batch_path = run_dir / "batch.json"
    if not batch_path.exists():
        generation_path = run_dir / "generation.json"
        if not generation_path.is_file():
            raise ValueError(
                f"Run directory contains neither batch.json nor generation.json: {run_dir}"
            )
        generation = _read_json(generation_path)
        if generation.get("success") is not False:
            return []
        device_id = request.device_ids[0] if request.device_ids else int(
            (generation.get("runtime_env") or {}).get("device_id", 0)
        )
        return [
            _task_from_generation(
                generation_path=generation_path,
                generation=generation,
                batch=None,
                op_record=None,
                candidate_record=None,
                device_id=device_id,
            )
        ]

    batch = _read_json(batch_path)
    op_records = batch.get("ops") or []
    if not isinstance(op_records, list):
        raise ValueError(f"batch.json field 'ops' must be a list: {batch_path}")

    tasks: list[FailedGenerationTask] = []
    override_devices = request.device_ids or []
    retry_index = 0
    for op_record in op_records:
        if not isinstance(op_record, dict):
            continue
        op_name = op_record.get("op_name")
        if not isinstance(op_name, str) or not op_name:
            continue
        op_dir = existing_op_run_dir(run_dir, op_name)
        candidate_dirs = sorted((op_dir / "candidates").glob("candidate_*"))
        candidate_records = _candidate_records_by_id(op_record)
        leaf_paths = (
            [path / "generation.json" for path in candidate_dirs]
            if candidate_dirs
            else [op_dir / "generation.json"]
        )
        for generation_path in leaf_paths:
            if not generation_path.is_file():
                continue
            generation = _read_json(generation_path)
            if generation.get("success") is not False:
                continue
            candidate_id = generation.get("candidate_id")
            if candidate_id is None:
                candidate_id = _candidate_id_from_dir(generation_path.parent)
            candidate_record = (
                candidate_records.get(int(candidate_id))
                if candidate_id is not None
                else None
            )
            if override_devices:
                device_id = override_devices[retry_index % len(override_devices)]
            else:
                device_id = int(
                    (candidate_record or {}).get(
                        "device_id",
                        op_record.get(
                            "device_id",
                            (generation.get("runtime_env") or {}).get("device_id", 0),
                        ),
                    )
                )
            tasks.append(
                _task_from_generation(
                    generation_path=generation_path,
                    generation=generation,
                    batch=batch,
                    op_record=op_record,
                    candidate_record=candidate_record,
                    device_id=device_id,
                )
            )
            retry_index += 1
    return tasks


def _run_retry_until_success(
    task: FailedGenerationTask,
    *,
    generation_retries: int,
    env_path: Path | None,
    log_fn: LogFn | None,
) -> tuple[dict, int]:
    start = time.monotonic()
    candidate_message = (
        f" candidate={task.candidate_id}"
        if task.candidate_id is not None
        else ""
    )
    if log_fn is not None:
        log_fn(
            "codegen",
            "start",
            op_name=task.op_name,
            device_id=task.device_id,
            message=f"retries={generation_retries}{candidate_message}",
        )

    initial_result = _read_json(task.generation_path)
    generation = run_generation_with_retries(
        GenerationRequest(
            run_id=task.run_id,
            op_name=task.op_name,
            shape_mode=task.shape_mode,
            shape_name=task.shape_name,
            shape_family=task.shape_family,
            language=task.language,
            run_dir=task.run_dir,
            device_id=task.device_id,
            env_path=env_path,
            no_overwrite=False,
            random_count=task.random_count,
            random_max_attempts=task.random_max_attempts,
            random_seed=task.random_seed,
            candidate_id=task.candidate_id,
            samples_per_op=task.samples_per_op,
            temperature=PASSK_TEMPERATURE if task.samples_per_op > 1 else None,
        ),
        generation_retries=generation_retries,
        initial_result=initial_result,
        on_retry=(
            lambda retry, total, _result: log_fn(
                "codegen",
                "retry",
                op_name=task.op_name,
                device_id=task.device_id,
                success=False,
                message=f"retry={retry}/{total}{candidate_message}",
            )
            if log_fn is not None
            else None
        ),
    )
    retry_meta = generation.get("generation_retry") or {}
    retries_used = int(retry_meta.get("retries_used") or 0)

    if log_fn is not None:
        log_fn(
            "codegen",
            "done",
            op_name=task.op_name,
            device_id=task.device_id,
            success=bool(generation.get("success")),
            elapsed=time.monotonic() - start,
            message=(
                f"retries_used={retries_used}/{generation_retries}"
                f"{candidate_message}"
            ),
        )
    return generation, retries_used


def _reward_hacking_risk_level(generation: dict) -> str | None:
    reward_hacking = generation.get("reward_hacking") or {}
    return reward_hacking.get("risk_level")


def _record_success(record: dict, generation_success: bool) -> bool:
    if "evaluation_success" in record:
        return generation_success and bool(record.get("evaluation_success"))
    if "compilation_success" in record:
        return generation_success and bool(record.get("compilation_success"))
    return generation_success


def _record_error(
    record: dict,
    generation: dict,
    generation_success: bool,
) -> tuple[object, object]:
    if not generation_success:
        return generation.get("error_type"), generation.get("error_message")
    if "evaluation_success" in record and not record.get("evaluation_success"):
        return record.get("error_type"), record.get("error_message")
    if "compilation_success" in record and not record.get("compilation_success"):
        return record.get("error_type"), record.get("error_message")
    return None, None


def _refresh_leaf_record(
    record: dict,
    *,
    generation: dict,
    generation_path: Path,
    op_name: str,
    device_id: int,
) -> dict:
    refreshed = dict(record)
    generation_success = bool(generation.get("success"))
    error_type, error_message = _record_error(
        refreshed,
        generation,
        generation_success,
    )
    refreshed.update(
        {
            "op_name": op_name,
            "category": get_operator_spec(op_name)["category"],
            "device_id": device_id,
            "run_dir": str(generation_path.parent),
            "generation_json": str(generation_path),
            "generated_path": str(
                generation_path.parent / "generated_ops" / f"{op_name}.txt"
            ),
            "reward_hacking_risk_level": _reward_hacking_risk_level(generation),
            "generation_success": generation_success,
            "success": _record_success(refreshed, generation_success),
            "error_type": error_type,
            "error_message": error_message,
        }
    )
    candidate_id = generation.get("candidate_id")
    if candidate_id is not None:
        refreshed["candidate_id"] = int(candidate_id)
        refreshed["candidate_name"] = f"candidate_{int(candidate_id):02d}"
    return refreshed


def _refresh_passk_op(
    *,
    batch: dict,
    op_record: dict,
    run_dir: Path,
    retried_generations: dict[Path, dict],
) -> dict:
    op_name = str(op_record["op_name"])
    category = get_operator_spec(op_name)["category"]
    op_dir = existing_op_run_dir(run_dir, op_name)
    summary_path = op_dir / "generation.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    old_candidates = _candidate_records_by_id(op_record)
    if not old_candidates:
        old_candidates = _candidate_records_by_id(summary)
    candidates = []
    candidate_dirs = sorted((op_dir / "candidates").glob("candidate_*"))
    for candidate_dir in candidate_dirs:
        generation_path = candidate_dir / "generation.json"
        candidate_id = _candidate_id_from_dir(candidate_dir)
        if candidate_id is None:
            continue
        old_record = old_candidates.get(candidate_id, {})
        generation = retried_generations.get(generation_path.resolve())
        if generation is None:
            candidates.append(old_record)
            continue
        device_id = int(
            (generation.get("runtime_env") or {}).get(
                "device_id",
                old_record.get("device_id", op_record.get("device_id", 0)),
            )
        )
        candidates.append(
            _refresh_leaf_record(
                old_record,
                generation=generation,
                generation_path=generation_path,
                op_name=op_name,
                device_id=device_id,
            )
        )
    candidates.sort(key=lambda item: int(item.get("candidate_id", 0)))
    successful_candidates = sum(
        1 for item in candidates if item.get("generation_success")
    )
    generation_success = successful_candidates > 0
    samples_per_op = int(
        op_record.get("samples_per_op")
        or batch.get("samples_per_op")
        or len(candidates)
    )

    summary.update(
        {
            "run_id": summary.get("run_id") or f"{batch.get('run_id')}_{op_name}",
            "op_name": op_name,
            "category": category,
            "shape_mode": summary.get("shape_mode") or batch.get("shape_mode"),
            "language": summary.get("language") or batch.get("language", "cuda"),
            "samples_per_op": samples_per_op,
            "temperature": PASSK_TEMPERATURE,
            "generation_success": generation_success,
            "success": generation_success,
            "total_candidates": len(candidates),
            "successful_candidates": successful_candidates,
            "failed_candidates": len(candidates) - successful_candidates,
            "candidates": candidates,
        }
    )
    if summary.get("shape_mode") == "single":
        summary["shape_name"] = summary.get("shape_name") or batch.get(
            "shape_name", "default"
        )
    elif summary.get("shape_mode") == "family":
        summary["shape_family"] = summary.get("shape_family") or batch.get(
            "shape_family", "all"
        )
    write_json(summary_path, summary)

    refreshed = dict(op_record)
    if generation_success:
        if "evaluation_success" in refreshed and not refreshed.get("evaluation_success"):
            error_type = refreshed.get("error_type")
            error_message = refreshed.get("error_message")
        elif "compilation_success" in refreshed and not refreshed.get(
            "compilation_success"
        ):
            error_type = refreshed.get("error_type")
            error_message = refreshed.get("error_message")
        else:
            error_type = None
            error_message = None
    else:
        error_type = "AllCandidatesFailed"
        error_message = "All pass@k candidates failed generation."
    refreshed.update(
        {
            "op_name": op_name,
            "category": category,
            "run_dir": str(op_dir),
            "generation_json": str(summary_path),
            "samples_per_op": samples_per_op,
            "total_candidates": len(candidates),
            "successful_candidates": successful_candidates,
            "candidates": candidates,
            "generation_success": generation_success,
            "success": _record_success(refreshed, generation_success),
            "error_type": error_type,
            "error_message": error_message,
        }
    )
    if "evaluation_success" not in refreshed and "compilation_success" not in refreshed:
        refreshed["failed_candidates"] = len(candidates) - successful_candidates
    return refreshed


def _refresh_batch(
    run_dir: Path,
    retried_generations: dict[Path, dict],
) -> dict | None:
    batch_path = run_dir / "batch.json"
    if not batch_path.is_file():
        return None
    batch = _read_json(batch_path)
    refreshed_ops = []
    for op_record in batch.get("ops") or []:
        if not isinstance(op_record, dict) or not op_record.get("op_name"):
            refreshed_ops.append(op_record)
            continue
        op_name = str(op_record["op_name"])
        op_dir = existing_op_run_dir(run_dir, op_name)
        if (op_dir / "candidates").is_dir():
            candidate_root = (op_dir / "candidates").resolve()
            has_retried_candidate = any(
                path.parent.parent == candidate_root
                for path in retried_generations
            )
            if not has_retried_candidate:
                refreshed_ops.append(op_record)
                continue
            refreshed_ops.append(
                _refresh_passk_op(
                    batch=batch,
                    op_record=op_record,
                    run_dir=run_dir,
                    retried_generations=retried_generations,
                )
            )
            continue
        generation_path = op_dir / "generation.json"
        generation = retried_generations.get(generation_path.resolve())
        if generation is None:
            refreshed_ops.append(op_record)
            continue
        device_id = int(
            (generation.get("runtime_env") or {}).get(
                "device_id", op_record.get("device_id", 0)
            )
        )
        refreshed_ops.append(
            _refresh_leaf_record(
                op_record,
                generation=generation,
                generation_path=generation_path,
                op_name=op_name,
                device_id=device_id,
            )
        )

    batch["ops"] = refreshed_ops
    generation_successful_ops = sum(
        1
        for record in refreshed_ops
        if isinstance(record, dict) and record.get("generation_success")
    )
    generation_success = generation_successful_ops == len(refreshed_ops)
    if "generation_success" in batch:
        batch["generation_success"] = generation_success
    if "evaluation_success" in batch:
        batch["success"] = generation_success and bool(batch.get("evaluation_success"))
        successful_ops = sum(
            1
            for record in refreshed_ops
            if isinstance(record, dict) and record.get("success")
        )
        batch["generation_successful_ops"] = generation_successful_ops
    elif "compilation_success" in batch:
        batch["success"] = generation_success and bool(batch.get("compilation_success"))
        successful_ops = sum(
            1
            for record in refreshed_ops
            if isinstance(record, dict) and record.get("success")
        )
        batch["generation_successful_ops"] = generation_successful_ops
    else:
        batch["success"] = generation_success
        successful_ops = generation_successful_ops
    batch["successful_ops"] = successful_ops
    batch["failed_ops"] = len(refreshed_ops) - successful_ops

    is_generation_only = (
        "evaluation_success" not in batch and "compilation_success" not in batch
    )
    if int(batch.get("samples_per_op") or 1) > 1 and is_generation_only:
        total_candidates = sum(
            int(record.get("total_candidates") or 0)
            for record in refreshed_ops
            if isinstance(record, dict)
        )
        successful_candidates = sum(
            int(record.get("successful_candidates") or 0)
            for record in refreshed_ops
            if isinstance(record, dict)
        )
        batch["total_candidates"] = total_candidates
        batch["successful_candidates"] = successful_candidates
        batch["failed_candidates"] = total_candidates - successful_candidates

    write_json(batch_path, batch)
    return batch


def retry_failed_generations(
    request: FailedGenerationRetryRequest,
    *,
    tasks: list[FailedGenerationTask] | None = None,
    log_fn: LogFn | None = None,
) -> dict:
    """Retry each failed generation until success or its retry limit."""
    if request.generation_retries < 0:
        raise ValueError("generation_retries must be >= 0.")
    run_dir = request.run_dir.resolve()
    retry_tasks = tasks if tasks is not None else discover_failed_generation_tasks(request)
    if request.generation_retries == 0:
        return {
            "success": not retry_tasks,
            "run_dir": str(run_dir),
            "retried_generations": 0,
            "successful_retries": 0,
            "failed_retries": len(retry_tasks),
            "generation_attempts": 0,
            "generation_retries": 0,
            "batch_success": None,
        }
    results: list[tuple[dict, int] | None] = [None] * len(retry_tasks)
    with ThreadPoolExecutor(max_workers=max(1, request.llm_workers)) as executor:
        futures = {
            executor.submit(
                _run_retry_until_success,
                task,
                generation_retries=request.generation_retries,
                env_path=request.env_path,
                log_fn=log_fn,
            ): index
            for index, task in enumerate(retry_tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()

    retried_generations = {
        task.generation_path.resolve(): result[0]
        for task, result in zip(retry_tasks, results)
        if result is not None
    }
    batch = _refresh_batch(run_dir, retried_generations)
    successful_retries = sum(
        1
        for result in results
        if result is not None and result[0].get("success")
    )
    retries_used = sum(
        result[1] for result in results if result is not None
    )
    return {
        "success": successful_retries == len(retry_tasks),
        "run_dir": str(run_dir),
        "retried_generations": len(retry_tasks),
        "successful_retries": successful_retries,
        "failed_retries": len(retry_tasks) - successful_retries,
        "generation_attempts": retries_used,
        "generation_retries": request.generation_retries,
        "batch_success": batch.get("success") if batch is not None else None,
    }
