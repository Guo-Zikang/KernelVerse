"""Batch scheduling for generated-code evaluation tasks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable

from evaluator.tasks import EvaluationTask
from evaluator.workflow import EvaluationRequest, evaluation_success, run_evaluation
from operators.registry import get_operator_spec
from runner.generated_code_checker import check_generated_file
from runner.generated_correctness_runner import (
    _case_subprocess_failure_result,
    finalize_generated_operator_result,
    prepare_generated_operator_evaluation,
    run_generated_operator_case,
    write_partial_generated_operator_result,
)
from runner.random_shape_config import resolve_random_shape_settings
from utils.run_paths import op_run_dir as category_op_run_dir


LogFn = Callable[..., None]
RUNNER_EXCEPTION_KEY = "__batch_runner_exception__"
EVAL_HEARTBEAT_INTERVAL_SECONDS = 1800.0


@dataclass
class _CaseWorkItem:
    op_index: int
    op_name: str
    language: str
    generated_path: Path
    op_run_dir: Path
    case_index: int
    raw_shape: dict
    trials: int
    perf_trials: int
    correctness_baseline_compute_mode: str


@dataclass
class _OpCaseState:
    index: int
    task: EvaluationTask
    op_run_dir: Path
    result: dict
    shapes: list[dict]
    random_correctness_config: dict
    correctness_baseline_compute_mode: str
    partial_metadata: dict
    total_cases: int
    completed_cases: int = 0
    devices_used: set[int] = field(default_factory=set)
    started_at: float = field(default_factory=time.monotonic)
    first_case_started_at: float | None = None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _reward_hacking_summary(reward_hacking_report: dict) -> dict:
    return {"risk_level": reward_hacking_report.get("risk_level", "low")}


def _torch_extensions_dir(op_run_dir: Path, device_id: int) -> Path:
    return op_run_dir / "torch_extensions" / f"device_{device_id}"


def _cleanup_torch_extensions(op_run_dir: Path) -> None:
    path = op_run_dir / "torch_extensions"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _resume_enabled(args: Any) -> bool:
    return bool(getattr(args, "resume", False))


def _read_resume_checkpoint(op_run_dir: Path) -> tuple[dict | None, str | None]:
    checkpoint_path = op_run_dir / "evaluation.json"
    if not checkpoint_path.exists():
        return None, None
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8")), None
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"Could not read resume checkpoint {checkpoint_path}: {exc}"


def _completed_resume_checkpoint(
    *,
    checkpoint: dict | None,
    task: EvaluationTask,
) -> dict | None:
    """Return a completed compatible checkpoint that needs no preparation."""
    if not checkpoint or checkpoint.get("status") != "done":
        return None
    if str(checkpoint.get("op_name")) != task.op_name:
        return None
    if str(checkpoint.get("shape_mode")) != task.shape_mode:
        return None
    if str(checkpoint.get("language")) != task.language:
        return None
    if not _same_path(checkpoint.get("generated_code"), task.generated_path):
        return None
    if task.candidate_id is not None:
        try:
            if int(checkpoint.get("candidate_id")) != int(task.candidate_id):
                return None
        except (TypeError, ValueError):
            return None

    try:
        total_cases = int(checkpoint["total_cases"])
        completed_cases = int(checkpoint["completed_cases"])
    except (KeyError, TypeError, ValueError):
        return None
    if total_cases < 0 or completed_cases != total_cases:
        return None

    cases = checkpoint.get("cases") or []
    if len(cases) != total_cases:
        return None
    completed_indices: set[int] = set()
    for case in cases:
        if not isinstance(case, dict) or case.get("case_index") is None:
            return None
        try:
            completed_indices.add(int(case["case_index"]))
        except (TypeError, ValueError):
            return None
    if completed_indices != set(range(total_cases)):
        return None
    return checkpoint


def _same_path(left: Any, right: Path) -> bool:
    if left is None:
        return False
    try:
        return Path(str(left)).resolve() == right.resolve()
    except Exception:
        return str(left) == str(right)


def _resume_cases_from_checkpoint(
    *,
    checkpoint: dict | None,
    task: EvaluationTask,
    shapes: list[dict],
) -> tuple[list[dict], str | None]:
    if not checkpoint:
        return [], None
    if str(checkpoint.get("op_name")) != task.op_name:
        return [], "checkpoint op_name does not match"
    if str(checkpoint.get("shape_mode")) != task.shape_mode:
        return [], "checkpoint shape_mode does not match"
    if str(checkpoint.get("language")) != task.language:
        return [], "checkpoint language does not match"
    if not _same_path(checkpoint.get("generated_code"), task.generated_path):
        return [], "checkpoint generated_code path does not match"

    total_cases = checkpoint.get("total_cases")
    if total_cases is not None and int(total_cases) != len(shapes):
        return [], "checkpoint total_cases does not match"

    cases_by_index: dict[int, dict] = {}
    for case in checkpoint.get("cases") or []:
        if not isinstance(case, dict) or case.get("case_index") is None:
            continue
        try:
            case_index = int(case["case_index"])
        except (TypeError, ValueError):
            continue
        if case_index < 0 or case_index >= len(shapes):
            continue
        if case.get("shape") != shapes[case_index]:
            continue
        cases_by_index[case_index] = case
    return [cases_by_index[index] for index in sorted(cases_by_index)], None


def _failed_trials_from_cases(cases: list[dict]) -> list[dict]:
    return [
        failure
        for case in cases
        for failure in (case.get("failed_trials") or [])
    ]


def _devices_from_cases(cases: list[dict]) -> set[int]:
    devices: set[int] = set()
    for case in cases:
        if case.get("device_id") is None:
            continue
        try:
            devices.add(int(case["device_id"]))
        except (TypeError, ValueError):
            continue
    return devices


def _candidate_dir(op_run_dir: Path, candidate_id: int) -> Path:
    return op_run_dir / "candidates" / f"candidate_{candidate_id:02d}"


def _task_run_dir(run_dir: Path, task: EvaluationTask) -> Path:
    op_run_dir = category_op_run_dir(run_dir, task.op_name)
    if task.candidate_id is None:
        return op_run_dir
    return _candidate_dir(op_run_dir, int(task.candidate_id))


def _run_case_with_elapsed(work_item: _CaseWorkItem, device_id: int) -> tuple[dict, float]:
    start = time.monotonic()
    payload = run_generated_operator_case(
        op_name=work_item.op_name,
        language=work_item.language,
        generated_path=str(work_item.generated_path),
        device_id=device_id,
        case_index=work_item.case_index,
        raw_shape=work_item.raw_shape,
        trials=work_item.trials,
        perf_trials=work_item.perf_trials,
        correctness_baseline_compute_mode=work_item.correctness_baseline_compute_mode,
        torch_extensions_dir=str(_torch_extensions_dir(work_item.op_run_dir, device_id)),
    )
    return payload, time.monotonic() - start


def batch_record_from_evaluation(
    op_name: str,
    device_id: int | None,
    op_run_dir: Path,
    result: dict,
    *,
    candidate_id: int | None = None,
    samples_per_op: int = 1,
) -> dict:
    category = get_operator_spec(op_name)["category"]
    error_message = result.get("error_message")
    correctness = result.get("correctness") or {}
    reward_hacking = result.get("reward_hacking") or {}
    result_success = evaluation_success(result)
    if error_message is None and not result_success and correctness.get("error_message"):
        error_message = correctness.get("error_message")
    if error_message is None and not result_success:
        error_message = "Evaluation failed."
    case_devices = sorted(
        {
            int(case["device_id"])
            for case in result.get("cases", [])
            if isinstance(case, dict) and case.get("device_id") is not None
        }
    )
    devices = case_devices or list(result.get("devices") or ([] if device_id is None else [device_id]))
    record = {
        "op_name": op_name,
        "category": category,
        "devices": devices,
        "run_dir": str(op_run_dir),
        "evaluation_json": str(op_run_dir / "evaluation.json"),
        "generated_path": result.get("generated_code"),
        "reward_hacking_risk_level": reward_hacking.get("risk_level"),
        "success": result_success,
        "error_type": result.get("error_type"),
        "error_message": error_message,
    }
    if candidate_id is not None:
        record["candidate_id"] = int(candidate_id)
        record["candidate_name"] = f"candidate_{int(candidate_id):02d}"
        record["samples_per_op"] = int(samples_per_op)
    return record


def _batch_record_for_task(
    task: EvaluationTask,
    device_id: int | None,
    op_run_dir: Path,
    result: dict,
) -> dict:
    return batch_record_from_evaluation(
        task.op_name,
        device_id,
        op_run_dir,
        result,
        candidate_id=task.candidate_id,
        samples_per_op=task.samples_per_op,
    )


def _has_candidate_records(records: list[dict]) -> bool:
    return any(record.get("candidate_id") is not None for record in records)


def _op_dir_from_candidate_record(record: dict) -> Path:
    run_dir = Path(str(record.get("run_dir")))
    if run_dir.name.startswith("candidate_") and run_dir.parent.name == "candidates":
        return run_dir.parent.parent
    return run_dir


def _passk_op_records(records: list[dict]) -> list[dict]:
    records_by_op: dict[str, list[dict]] = {}
    for record in records:
        records_by_op.setdefault(str(record.get("op_name")), []).append(record)

    op_records = []
    for op_name, candidate_records in records_by_op.items():
        ordered_candidates = sorted(
            candidate_records,
            key=lambda item: int(item.get("candidate_id", 0)),
        )
        passed_candidates = sum(1 for item in ordered_candidates if item.get("success"))
        first_record = ordered_candidates[0]
        op_run_dir = _op_dir_from_candidate_record(first_record)
        samples_per_op = max(
            int(item.get("samples_per_op") or len(ordered_candidates))
            for item in ordered_candidates
        )
        pass_at_k = passed_candidates > 0
        error_message = None if pass_at_k else "No candidate passed evaluation."
        op_records.append(
            {
                "op_name": op_name,
                "category": first_record.get("category"),
                "devices": sorted(
                    {
                        int(device)
                        for item in ordered_candidates
                        for device in (item.get("devices") or [])
                    }
                ),
                "run_dir": str(op_run_dir),
                "evaluation_json": str(op_run_dir / "evaluation.json"),
                "samples_per_op": samples_per_op,
                "total_candidates": len(ordered_candidates),
                "passed_candidates": passed_candidates,
                "failed_candidates": len(ordered_candidates) - passed_candidates,
                "pass_at_10_raw": pass_at_k,
                "success": pass_at_k,
                "error_type": None if pass_at_k else "NoPassingCandidate",
                "error_message": error_message,
                "candidates": ordered_candidates,
            }
        )
    return sorted(op_records, key=lambda item: str(item.get("op_name")))


def _summary_payload(
    *,
    records: list[dict],
    summary_context: dict[str, Any],
    run_dir: Path,
    device_ids: list[int],
    category: str | None = None,
) -> dict:
    if _has_candidate_records(records):
        op_records = _passk_op_records(records)
        total_candidates = sum(int(record.get("total_candidates") or 0) for record in op_records)
        passed_candidates = sum(int(record.get("passed_candidates") or 0) for record in op_records)
        pass_at_10_raw_ops = sum(1 for record in op_records if record.get("pass_at_10_raw"))
        payload = {
            "success": all(record["success"] for record in op_records),
            **summary_context,
            "run_dir": str(run_dir),
            "device_ids": device_ids,
            "samples_per_op": max(
                int(record.get("samples_per_op") or 1)
                for record in op_records
            ) if op_records else 1,
            "total_ops": len(op_records),
            "successful_ops": pass_at_10_raw_ops,
            "failed_ops": len(op_records) - pass_at_10_raw_ops,
            "pass_at_10_raw_ops": pass_at_10_raw_ops,
            "pass_at_10_raw_rate": (
                round(pass_at_10_raw_ops / len(op_records), 4)
                if op_records
                else 0.0
            ),
            "total_candidates": total_candidates,
            "passed_candidates": passed_candidates,
            "failed_candidates": total_candidates - passed_candidates,
            "ops": op_records,
        }
        if category is not None:
            payload["category"] = category
        return payload

    payload = {
        "success": all(record["success"] for record in records),
        **summary_context,
        "run_dir": str(run_dir),
        "device_ids": device_ids,
        "total_ops": len(records),
        "successful_ops": sum(1 for record in records if record["success"]),
        "failed_ops": sum(1 for record in records if not record["success"]),
        "ops": records,
    }
    if category is not None:
        payload["category"] = category
    return payload


def _write_passk_op_summaries(records: list[dict]) -> None:
    if not _has_candidate_records(records):
        return
    for record in _passk_op_records(records):
        write_json(Path(str(record["evaluation_json"])), record)


def _write_category_summaries(
    *,
    run_dir: Path,
    records: list[dict],
    summary_context: dict[str, Any],
    device_ids: list[int],
) -> None:
    records_by_category: dict[str, list[dict]] = {}
    for record in records:
        category = str(record.get("category") or get_operator_spec(record["op_name"])["category"])
        records_by_category.setdefault(category, []).append(record)

    for category, category_records in records_by_category.items():
        category_dir = run_dir / "ops" / category
        payload = _summary_payload(
            records=category_records,
            summary_context=summary_context,
            run_dir=category_dir,
            device_ids=device_ids,
            category=category,
        )
        write_json(category_dir / "evaluation.json", payload)


def _exception_result(
    *,
    task: EvaluationTask,
    run_dir: Path,
    devices: list[int] | None = None,
    verify_trials: int,
    perf_trials: int,
    error: Exception,
    error_type: str | None = None,
) -> dict:
    result = {
        "op_name": task.op_name,
        "shape_mode": task.shape_mode,
        "language": task.language,
        "devices": list(devices or []),
        "run_id": f"{run_dir.name}_{task.op_name}",
        "perf_trials_per_shape": perf_trials,
        "generated_code": str(task.generated_path),
        "reward_hacking": {
            "risk_level": "low",
        },
        "compiled": None,
        "correctness": {
            "success": False,
            "num_trials": verify_trials,
            "failed_trials": [],
            "error_message": str(error),
        },
        "cases": [],
        "error_type": error_type or type(error).__name__,
        "error_message": str(error),
    }
    if task.candidate_id is not None:
        result["candidate_id"] = int(task.candidate_id)
        result["samples_per_op"] = int(task.samples_per_op)
    return result


def _safe_write_json(path: Path, payload: dict) -> None:
    try:
        write_json(path, payload)
    except Exception:
        pass


def _safe_log_done(
    log_fn: LogFn | None,
    *,
    op_name: str,
    device_id: int | None,
    success: bool,
    elapsed: float,
) -> None:
    if log_fn is None:
        return
    try:
        log_fn(
            "eval",
            "done",
            op_name=op_name,
            device_id=device_id,
            success=success,
            elapsed=elapsed,
        )
    except Exception:
        pass


def _safe_operator_category(op_name: str) -> str | None:
    try:
        return get_operator_spec(op_name)["category"]
    except Exception:
        return None


def _run_evaluation_with_elapsed(request: EvaluationRequest) -> tuple[dict, float]:
    """Run one evaluation task and measure actual child-process execution time."""
    start = time.monotonic()
    try:
        return run_evaluation(request), time.monotonic() - start
    except Exception as exc:  # pylint: disable=broad-except
        return (
            {
                RUNNER_EXCEPTION_KEY: True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            time.monotonic() - start,
        )


class StreamingEvaluationScheduler:
    """Submit generated-code evaluations as soon as their code is available."""

    def __init__(
        self,
        *,
        args: Any,
        run_dir: Path,
        summary_context: dict[str, Any],
        log_fn: LogFn | None = None,
    ) -> None:
        self.args = args
        self.run_dir = run_dir
        self.summary_context = summary_context
        self.log_fn = log_fn
        self._records: list[dict | None] = []
        self._futures = {}
        self._device_queues: dict[int, list[tuple[int, EvaluationTask, Path]]] = {
            device_id: [] for device_id in args.device_ids
        }
        self._active_by_device: dict[int, dict[str, Any]] = {}
        self._condition = threading.Condition()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="eval-heartbeat",
            daemon=True,
        )
        spawn_context = mp.get_context("spawn")
        self._executors = {
            device_id: ProcessPoolExecutor(max_workers=1, mp_context=spawn_context)
            for device_id in args.device_ids
        }
        self._heartbeat_thread.start()

    def _activate_next_locked(self, device_id: int) -> dict[str, Any] | None:
        if device_id in self._active_by_device:
            return None
        queue = self._device_queues.get(device_id, [])
        if not queue:
            return None
        index, task, op_run_dir = queue[0]
        self._active_by_device[device_id] = {
            "index": index,
            "task": task,
            "op_run_dir": op_run_dir,
            "started_at": time.monotonic(),
        }
        return {
            "op_name": task.op_name,
            "device_id": device_id,
            "queued_on_device": len(queue),
        }

    def _complete_active_locked(self, device_id: int, index: int) -> dict[str, Any] | None:
        queue = self._device_queues.get(device_id, [])
        if queue and queue[0][0] == index:
            queue.pop(0)
        else:
            self._device_queues[device_id] = [item for item in queue if item[0] != index]

        active_state = self._active_by_device.get(device_id)
        if active_state is not None and active_state.get("index") == index:
            self._active_by_device.pop(device_id, None)
        return self._activate_next_locked(device_id)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(EVAL_HEARTBEAT_INTERVAL_SECONDS):
            events = []
            with self._condition:
                now = time.monotonic()
                for device_id, active_state in sorted(self._active_by_device.items()):
                    task = active_state["task"]
                    queue = self._device_queues.get(device_id, [])
                    events.append(
                        {
                            "op_name": task.op_name,
                            "device_id": device_id,
                            "elapsed": now - active_state["started_at"],
                            "queued_on_device": len(queue),
                        }
                )

            if self.log_fn is None:
                continue
            for event in events:
                self.log_fn(
                    "eval",
                    "running",
                    op_name=event["op_name"],
                    device_id=event["device_id"],
                    elapsed=event["elapsed"],
                    message=f"queued_on_device={event['queued_on_device']}",
                )

    def _log_active(self, active_event: dict[str, Any] | None) -> None:
        if self.log_fn is None or active_event is None:
            return
        self.log_fn(
            "eval",
            "active",
            op_name=active_event["op_name"],
            device_id=active_event["device_id"],
            message=f"queued_on_device={active_event['queued_on_device']}",
        )

    def submit(self, task: EvaluationTask, *, device_id: int, op_run_dir: Path) -> None:
        with self._condition:
            index = len(self._records)
            self._records.append(None)
            self._device_queues.setdefault(device_id, []).append((index, task, op_run_dir))
            active_event = self._activate_next_locked(device_id)

        if self.log_fn is not None:
            self.log_fn("eval", "queued", op_name=task.op_name, device_id=device_id)
        self._log_active(active_event)

        request = EvaluationRequest(
            run_id=f"{self.run_dir.name}_{task.op_name}",
            op_name=task.op_name,
            shape_mode=task.shape_mode,
            shape_name=task.shape_name,
            shape_family=task.shape_family,
            language=task.language,
            generated_path=task.generated_path,
            run_dir=op_run_dir,
            device_ids=[device_id],
            verify_trials=self.args.verify_trials,
            perf_trials=self.args.perf_trials,
            random_count=task.random_count,
            random_max_attempts=task.random_max_attempts,
        )
        future = self._executors[device_id].submit(_run_evaluation_with_elapsed, request)
        with self._condition:
            self._futures[future] = (index, task, device_id, op_run_dir)
        future.add_done_callback(self._complete_future)

    def record_failure(
        self,
        task: EvaluationTask,
        *,
        device_id: int,
        op_run_dir: Path,
        error: Exception,
        error_type: str | None = None,
    ) -> None:
        result = _exception_result(
            task=task,
            run_dir=self.run_dir,
            verify_trials=self.args.verify_trials,
            perf_trials=self.args.perf_trials,
            error=error,
            error_type=error_type,
        )
        write_json(op_run_dir / "evaluation.json", result)
        _cleanup_torch_extensions(op_run_dir)
        record = _batch_record_for_task(task, device_id, op_run_dir, result)
        with self._condition:
            self._records.append(record)
            self._condition.notify_all()

    def _op_level_callback_failure(self, payload, error: Exception) -> None:
        if payload is None:
            return
        index, task, device_id, op_run_dir = payload
        result = _exception_result(
            task=task,
            run_dir=self.run_dir,
            verify_trials=self.args.verify_trials,
            perf_trials=self.args.perf_trials,
            error=error,
            error_type="EvaluationCallbackFailure",
        )
        _safe_write_json(op_run_dir / "evaluation.json", result)
        _cleanup_torch_extensions(op_run_dir)
        try:
            record = _batch_record_for_task(task, device_id, op_run_dir, result)
        except Exception:
            record = {
                "op_name": task.op_name,
                "category": _safe_operator_category(task.op_name),
                "devices": [device_id],
                "run_dir": str(op_run_dir),
                "evaluation_json": str(op_run_dir / "evaluation.json"),
                "generated_path": str(task.generated_path),
                "success": False,
                "error_type": "EvaluationCallbackFailure",
                "error_message": str(error),
            }
            if task.candidate_id is not None:
                record["candidate_id"] = int(task.candidate_id)
                record["candidate_name"] = f"candidate_{int(task.candidate_id):02d}"
                record["samples_per_op"] = int(task.samples_per_op)
        with self._condition:
            self._complete_active_locked(device_id, index)
            self._records[index] = record
            self._condition.notify_all()
        _safe_log_done(
            self.log_fn,
            op_name=task.op_name,
            device_id=device_id,
            success=False,
            elapsed=0.0,
        )

    def _complete_future(self, future) -> None:
        with self._condition:
            payload = self._futures.pop(future, None)
        if payload is None:
            return
        try:
            self._complete_future_impl(future, payload)
        except Exception as exc:  # pylint: disable=broad-except
            self._op_level_callback_failure(payload, exc)

    def _complete_future_impl(self, future, payload) -> None:
        index, task, device_id, op_run_dir = payload
        elapsed = 0.0
        try:
            result, elapsed = future.result()
            if result.pop(RUNNER_EXCEPTION_KEY, False):
                error_message = result.get("error_message") or "Evaluation failed."
                result = _exception_result(
                    task=task,
                    run_dir=self.run_dir,
                    verify_trials=self.args.verify_trials,
                    perf_trials=self.args.perf_trials,
                    error=RuntimeError(error_message),
                    error_type=result.get("error_type"),
                )
                write_json(op_run_dir / "evaluation.json", result)
                _cleanup_torch_extensions(op_run_dir)
        except Exception as exc:  # pylint: disable=broad-except
            result = _exception_result(
                task=task,
                run_dir=self.run_dir,
                verify_trials=self.args.verify_trials,
                perf_trials=self.args.perf_trials,
                error=exc,
            )
            write_json(op_run_dir / "evaluation.json", result)
            _cleanup_torch_extensions(op_run_dir)

        if self.log_fn is not None:
            self.log_fn(
                "eval",
                "done",
                op_name=task.op_name,
                device_id=device_id,
                success=evaluation_success(result),
                elapsed=elapsed,
            )
        record = _batch_record_for_task(task, device_id, op_run_dir, result)
        active_event = None
        with self._condition:
            active_event = self._complete_active_locked(device_id, index)
            self._records[index] = record
            self._condition.notify_all()
        self._log_active(active_event)

    def finish(self, *, write_summary: bool = True) -> dict:
        try:
            for executor in self._executors.values():
                executor.shutdown(wait=True)

            with self._condition:
                while any(record is None for record in self._records):
                    self._condition.wait(timeout=0.1)
                finalized = [record for record in self._records if record is not None]
        finally:
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=1.0)

        payload = _summary_payload(
            records=finalized,
            summary_context=self.summary_context,
            run_dir=self.run_dir,
            device_ids=self.args.device_ids,
        )
        if write_summary:
            _write_passk_op_summaries(finalized)
            _write_category_summaries(
                run_dir=self.run_dir,
                records=finalized,
                summary_context=self.summary_context,
                device_ids=self.args.device_ids,
            )
            write_json(self.run_dir / "evaluation.json", payload)
        return payload


class CaseLevelStreamingEvaluationScheduler:
    """Schedule generated-code evaluation at case granularity across devices."""

    def __init__(
        self,
        *,
        args: Any,
        run_dir: Path,
        summary_context: dict[str, Any],
        log_fn: LogFn | None = None,
    ) -> None:
        self.args = args
        self.run_dir = run_dir
        self.summary_context = summary_context
        self.log_fn = log_fn
        self._records: list[dict | None] = []
        self._prepare_futures = {}
        self._case_futures = {}
        self._case_queue: list[_CaseWorkItem] = []
        self._op_states: dict[int, _OpCaseState] = {}
        self._active_by_device: dict[int, dict[str, Any]] = {}
        self._condition = threading.Condition()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="eval-heartbeat",
            daemon=True,
        )
        prepare_workers = max(1, min(len(args.device_ids), 4))
        self._prepare_executor = ThreadPoolExecutor(max_workers=prepare_workers)
        self._executors = {
            device_id: ThreadPoolExecutor(max_workers=1)
            for device_id in args.device_ids
        }
        self._heartbeat_thread.start()

    def _partial_path(self, state: _OpCaseState) -> Path:
        return state.op_run_dir / "evaluation.json"

    def _running_cases_for_state_locked(self, op_index: int) -> list[dict[str, Any]]:
        now = time.monotonic()
        running_cases = []
        for device_id, active_state in sorted(self._active_by_device.items()):
            work_item = active_state["work_item"]
            if work_item.op_index != op_index:
                continue
            running_cases.append(
                {
                    "case_index": work_item.case_index,
                    "device_id": device_id,
                    "elapsed": round(now - active_state["started_at"], 1),
                }
            )
        return running_cases

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(EVAL_HEARTBEAT_INTERVAL_SECONDS):
            events = []
            with self._condition:
                now = time.monotonic()
                for device_id, active_state in sorted(self._active_by_device.items()):
                    work_item = active_state["work_item"]
                    events.append(
                        {
                            "op_name": work_item.op_name,
                            "case_index": work_item.case_index,
                            "device_id": device_id,
                            "elapsed": now - active_state["started_at"],
                            "queued_cases": len(self._case_queue),
                        }
                    )
            if self.log_fn is None:
                continue
            for event in events:
                self.log_fn(
                    "eval",
                    "running",
                    op_name=event["op_name"],
                    device_id=event["device_id"],
                    elapsed=event["elapsed"],
                    message=(
                        f"case={event['case_index']} "
                        f"queued_cases={event['queued_cases']}"
                    ),
                )

    def _log_case_active(self, work_item: _CaseWorkItem, device_id: int) -> None:
        if self.log_fn is None:
            return
        self.log_fn(
            "eval",
            "active",
            op_name=work_item.op_name,
            device_id=device_id,
            message=f"case={work_item.case_index}",
        )

    def _schedule_cases_locked(self) -> None:
        for device_id, executor in self._executors.items():
            if device_id in self._active_by_device or not self._case_queue:
                continue
            work_item = self._case_queue.pop(0)
            now = time.monotonic()
            op_state = self._op_states.get(work_item.op_index)
            if op_state is not None and op_state.first_case_started_at is None:
                op_state.first_case_started_at = now
            self._active_by_device[device_id] = {
                "work_item": work_item,
                "started_at": now,
            }
            future = executor.submit(_run_case_with_elapsed, work_item, device_id)
            self._case_futures[future] = (work_item, device_id)
            future.add_done_callback(self._complete_case_future)
            self._log_case_active(work_item, device_id)

    def _prepare_op(
        self,
        *,
        index: int,
        task: EvaluationTask,
        device_id: int,
        op_run_dir: Path,
    ) -> dict:
        resume_checkpoint = None
        resume_warning = None
        if _resume_enabled(self.args):
            resume_checkpoint, resume_warning = _read_resume_checkpoint(op_run_dir)
            completed_checkpoint = _completed_resume_checkpoint(
                checkpoint=resume_checkpoint,
                task=task,
            )
            if completed_checkpoint is not None:
                return {
                    "completed_checkpoint": completed_checkpoint,
                }

        reward_hacking_report = check_generated_file(task.generated_path, op_name=task.op_name)
        reward_hacking = _reward_hacking_summary(reward_hacking_report)
        write_json(op_run_dir / "reward_hacking_report.json", reward_hacking_report)
        partial_metadata = {
            "run_id": f"{self.run_dir.name}_{task.op_name}",
            "reward_hacking": reward_hacking,
        }
        if task.candidate_id is not None:
            partial_metadata["candidate_id"] = int(task.candidate_id)
            partial_metadata["samples_per_op"] = int(task.samples_per_op)
        try:
            operator_spec = get_operator_spec(task.op_name)
            random_count, random_max_attempts = resolve_random_shape_settings(
                operator_spec,
                random_count=task.random_count,
                random_max_attempts=task.random_max_attempts,
            )
            plan = prepare_generated_operator_evaluation(
                op_name=task.op_name,
                language=task.language,
                shape_mode=task.shape_mode,
                shape_name=task.shape_name,
                shape_family=task.shape_family,
                generated_path=str(task.generated_path),
                compile_device_id=device_id,
                devices=list(self.args.device_ids),
                trials=self.args.verify_trials,
                random_seed=getattr(self.args, "random_seed", 42),
                random_count=random_count,
                random_max_attempts=random_max_attempts,
                perf_trials=self.args.perf_trials,
                partial_result_path=(
                    None
                    if resume_checkpoint is not None
                    else str(op_run_dir / "evaluation.json")
                ),
                partial_result_metadata=partial_metadata,
                torch_extensions_dir=str(_torch_extensions_dir(op_run_dir, device_id)),
            )
            plan["result"]["run_id"] = partial_metadata["run_id"]
            plan["result"]["reward_hacking"] = reward_hacking
            if task.candidate_id is not None:
                plan["result"]["candidate_id"] = int(task.candidate_id)
                plan["result"]["samples_per_op"] = int(task.samples_per_op)
            return {
                "plan": plan,
                "partial_metadata": partial_metadata,
                "resume_checkpoint": resume_checkpoint,
                "resume_warning": resume_warning,
            }
        except Exception as exc:  # pylint: disable=broad-except
            result = _exception_result(
                task=task,
                run_dir=self.run_dir,
                devices=list(self.args.device_ids),
                verify_trials=self.args.verify_trials,
                perf_trials=self.args.perf_trials,
                error=exc,
            )
            result["reward_hacking"] = reward_hacking
            write_json(op_run_dir / "evaluation.json", result)
            return {
                "plan": {
                    "prepared": False,
                    "result": result,
                    "shapes": [],
                    "correctness_baseline_compute_mode": None,
                    "random_correctness_config": {"max_failed_case_rate": 0.03},
                },
                "partial_metadata": partial_metadata,
                "resume_checkpoint": resume_checkpoint,
                "resume_warning": resume_warning,
            }

    def submit(self, task: EvaluationTask, *, device_id: int, op_run_dir: Path) -> None:
        with self._condition:
            index = len(self._records)
            self._records.append(None)
        if self.log_fn is not None:
            self.log_fn("eval", "queued", op_name=task.op_name, device_id=device_id)
        future = self._prepare_executor.submit(
            self._prepare_op,
            index=index,
            task=task,
            device_id=device_id,
            op_run_dir=op_run_dir,
        )
        with self._condition:
            self._prepare_futures[future] = (index, task, device_id, op_run_dir)
        future.add_done_callback(self._complete_prepare_future)

    def record_failure(
        self,
        task: EvaluationTask,
        *,
        device_id: int,
        op_run_dir: Path,
        error: Exception,
        error_type: str | None = None,
    ) -> None:
        result = _exception_result(
            task=task,
            run_dir=self.run_dir,
            devices=[device_id],
            verify_trials=self.args.verify_trials,
            perf_trials=self.args.perf_trials,
            error=error,
            error_type=error_type,
        )
        write_json(op_run_dir / "evaluation.json", result)
        record = _batch_record_for_task(task, device_id, op_run_dir, result)
        with self._condition:
            self._records.append(record)
            self._condition.notify_all()

    def _prepare_callback_failure(self, payload, error: Exception) -> None:
        if payload is None:
            return
        index, task, device_id, op_run_dir = payload
        result = _exception_result(
            task=task,
            run_dir=self.run_dir,
            devices=list(self.args.device_ids),
            verify_trials=self.args.verify_trials,
            perf_trials=self.args.perf_trials,
            error=error,
            error_type="PrepareCallbackFailure",
        )
        _safe_write_json(op_run_dir / "evaluation.json", result)
        _cleanup_torch_extensions(op_run_dir)
        try:
            record = _batch_record_for_task(task, device_id, op_run_dir, result)
        except Exception:
            record = {
                "op_name": task.op_name,
                "category": _safe_operator_category(task.op_name),
                "devices": [device_id],
                "run_dir": str(op_run_dir),
                "evaluation_json": str(op_run_dir / "evaluation.json"),
                "generated_path": str(task.generated_path),
                "success": False,
                "error_type": "PrepareCallbackFailure",
                "error_message": str(error),
            }
            if task.candidate_id is not None:
                record["candidate_id"] = int(task.candidate_id)
                record["candidate_name"] = f"candidate_{int(task.candidate_id):02d}"
                record["samples_per_op"] = int(task.samples_per_op)
        _safe_log_done(
            self.log_fn,
            op_name=task.op_name,
            device_id=device_id,
            success=False,
            elapsed=0.0,
        )
        with self._condition:
            self._records[index] = record
            self._condition.notify_all()

    def _complete_prepare_future(self, future) -> None:
        with self._condition:
            payload = self._prepare_futures.pop(future, None)
        if payload is None:
            return
        try:
            self._complete_prepare_future_impl(future, payload)
        except Exception as exc:  # pylint: disable=broad-except
            self._prepare_callback_failure(payload, exc)

    def _complete_prepare_future_impl(self, future, payload) -> None:
        index, task, device_id, op_run_dir = payload
        try:
            payload = future.result()
            completed_checkpoint = payload.get("completed_checkpoint")
            if completed_checkpoint is not None:
                record = _batch_record_for_task(
                    task,
                    None,
                    op_run_dir,
                    completed_checkpoint,
                )
                if self.log_fn is not None:
                    self.log_fn(
                        "eval",
                        "resume",
                        op_name=task.op_name,
                        message=(
                            "completed checkpoint reused; "
                            f"completed_cases={completed_checkpoint['completed_cases']} "
                            "pending_cases=0"
                        ),
                    )
                    self.log_fn(
                        "eval",
                        "done",
                        op_name=task.op_name,
                        device_id=device_id,
                        success=evaluation_success(completed_checkpoint),
                        elapsed=0.0,
                    )
                with self._condition:
                    self._records[index] = record
                    self._condition.notify_all()
                return
            plan = payload["plan"]
            partial_metadata = payload["partial_metadata"]
            result = plan["result"]
            resume_checkpoint = payload.get("resume_checkpoint")
            resume_warning = payload.get("resume_warning")
        except Exception as exc:  # pylint: disable=broad-except
            result = _exception_result(
                task=task,
                run_dir=self.run_dir,
                devices=list(self.args.device_ids),
                verify_trials=self.args.verify_trials,
                perf_trials=self.args.perf_trials,
                error=exc,
            )
            write_json(op_run_dir / "evaluation.json", result)
            plan = {"prepared": False, "result": result, "shapes": []}
            partial_metadata = {}
            resume_checkpoint = None
            resume_warning = None

        if not plan.get("prepared"):
            _cleanup_torch_extensions(op_run_dir)
            write_json(op_run_dir / "evaluation.json", result)
            if self.log_fn is not None:
                self.log_fn(
                    "eval",
                    "done",
                    op_name=task.op_name,
                    device_id=device_id,
                    success=evaluation_success(result),
                    elapsed=0.0,
                )
            record = _batch_record_for_task(task, device_id, op_run_dir, result)
            with self._condition:
                self._records[index] = record
                self._condition.notify_all()
            return

        shapes = list(plan["shapes"])
        resumed_cases, resume_ignore_reason = _resume_cases_from_checkpoint(
            checkpoint=resume_checkpoint,
            task=task,
            shapes=shapes,
        )
        if resume_warning and self.log_fn is not None:
            self.log_fn("eval", "resume", op_name=task.op_name, message=resume_warning)
        if resume_ignore_reason and self.log_fn is not None:
            self.log_fn(
                "eval",
                "resume",
                op_name=task.op_name,
                message=f"checkpoint ignored: {resume_ignore_reason}",
            )
        resumed_indices = {int(case["case_index"]) for case in resumed_cases}
        if resumed_cases:
            result["cases"] = list(resumed_cases)
            result["correctness"]["failed_trials"] = _failed_trials_from_cases(resumed_cases)

        state = _OpCaseState(
            index=index,
            task=task,
            op_run_dir=op_run_dir,
            result=result,
            shapes=shapes,
            random_correctness_config=plan["random_correctness_config"],
            correctness_baseline_compute_mode=plan["correctness_baseline_compute_mode"],
            partial_metadata=partial_metadata,
            total_cases=len(shapes),
            completed_cases=len(resumed_cases),
            devices_used=_devices_from_cases(resumed_cases),
        )
        case_items = [
            _CaseWorkItem(
                op_index=index,
                op_name=task.op_name,
                language=task.language,
                generated_path=task.generated_path,
                op_run_dir=op_run_dir,
                case_index=case_index,
                raw_shape=raw_shape,
                trials=self.args.verify_trials,
                perf_trials=self.args.perf_trials,
                correctness_baseline_compute_mode=state.correctness_baseline_compute_mode,
            )
            for case_index, raw_shape in enumerate(shapes)
            if case_index not in resumed_indices
        ]
        if _resume_enabled(self.args) and self.log_fn is not None:
            self.log_fn(
                "eval",
                "resume",
                op_name=task.op_name,
                message=(
                    f"completed_cases={len(resumed_cases)} "
                    f"pending_cases={len(case_items)}"
                ),
            )
        if not case_items:
            finalize_generated_operator_result(
                result=result,
                shape_mode=task.shape_mode,
                random_correctness_config=plan["random_correctness_config"],
                total_cases=len(shapes),
            )
            result["devices"] = sorted(state.devices_used) or list(self.args.device_ids)
            write_partial_generated_operator_result(
                partial_result_path=op_run_dir / "evaluation.json",
                result=result,
                status="done",
                partial_result_metadata=partial_metadata,
                total_cases=len(shapes),
                completed_cases=len(resumed_cases),
                last_completed_case_index=max(resumed_indices) if resumed_indices else None,
                running_cases=[],
            )
            _cleanup_torch_extensions(op_run_dir)
            record = _batch_record_for_task(task, None, op_run_dir, result)
            if self.log_fn is not None:
                self.log_fn(
                    "eval",
                    "done",
                    op_name=task.op_name,
                    device_id=device_id,
                    success=evaluation_success(result),
                    elapsed=0.0,
                )
            with self._condition:
                self._records[index] = record
                self._condition.notify_all()
            return
        with self._condition:
            self._op_states[index] = state
            self._case_queue.extend(case_items)
            write_partial_generated_operator_result(
                partial_result_path=op_run_dir / "evaluation.json",
                result=state.result,
                status="running",
                partial_result_metadata=partial_metadata,
                total_cases=state.total_cases,
                completed_cases=state.completed_cases,
                last_completed_case_index=max(resumed_indices) if resumed_indices else None,
                running_cases=[],
            )
            self._schedule_cases_locked()
            self._condition.notify_all()

    def _case_callback_failure(self, payload, error: Exception) -> None:
        if payload is None:
            return
        work_item, device_id = payload
        fallback_case = _case_subprocess_failure_result(
            raw_shape=work_item.raw_shape,
            case_index=work_item.case_index,
            error_type="CaseCallbackFailure",
            error_message=str(error),
            device_id=device_id,
        )["case_result"]
        completed_record = None
        completed_log = None
        with self._condition:
            self._active_by_device.pop(device_id, None)
            state = self._op_states.get(work_item.op_index)
            if state is None:
                self._condition.notify_all()
                return
            state.result["cases"].append(fallback_case)
            state.completed_cases += 1
            state.devices_used.add(device_id)
            state.result["correctness"]["failed_trials"] = _failed_trials_from_cases(
                state.result.get("cases") or []
            )
            state.result["error_type"] = "CaseCallbackFailure"
            state.result["error_message"] = str(error)
            try:
                finalize_generated_operator_result(
                    result=state.result,
                    shape_mode=state.task.shape_mode,
                    random_correctness_config=state.random_correctness_config,
                    total_cases=state.total_cases,
                )
            except Exception:
                state.result["correctness"]["success"] = False
            state.result["devices"] = sorted(state.devices_used) or list(self.args.device_ids)
            _safe_write_json(self._partial_path(state), state.result)
            _cleanup_torch_extensions(state.op_run_dir)
            try:
                completed_record = _batch_record_for_task(
                    state.task,
                    None,
                    state.op_run_dir,
                    state.result,
                )
            except Exception:
                completed_record = {
                    "op_name": state.task.op_name,
                    "category": _safe_operator_category(state.task.op_name),
                    "devices": sorted(state.devices_used) or [device_id],
                    "run_dir": str(state.op_run_dir),
                    "evaluation_json": str(state.op_run_dir / "evaluation.json"),
                    "generated_path": str(state.task.generated_path),
                    "success": False,
                    "error_type": "CaseCallbackFailure",
                    "error_message": str(error),
                }
                if state.task.candidate_id is not None:
                    completed_record["candidate_id"] = int(state.task.candidate_id)
                    completed_record["candidate_name"] = f"candidate_{int(state.task.candidate_id):02d}"
                    completed_record["samples_per_op"] = int(state.task.samples_per_op)
            completed_log = {
                "op_name": state.task.op_name,
                "success": False,
                "elapsed": time.monotonic() - (state.first_case_started_at or state.started_at),
            }
            self._records[state.index] = completed_record
            self._op_states.pop(state.index, None)
            self._schedule_cases_locked()
            self._condition.notify_all()

        if completed_log is not None:
            _safe_log_done(
                self.log_fn,
                op_name=completed_log["op_name"],
                device_id=device_id,
                success=completed_log["success"],
                elapsed=completed_log["elapsed"],
            )

    def _complete_case_future(self, future) -> None:
        with self._condition:
            payload = self._case_futures.pop(future, None)
        if payload is None:
            return
        try:
            self._complete_case_future_impl(future, payload)
        except Exception as exc:  # pylint: disable=broad-except
            self._case_callback_failure(payload, exc)

    def _complete_case_future_impl(self, future, payload) -> None:
        work_item, device_id = payload
        with self._condition:
            self._active_by_device.pop(device_id, None)
        elapsed = 0.0
        try:
            payload, elapsed = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            payload = _case_subprocess_failure_result(
                raw_shape=work_item.raw_shape,
                case_index=work_item.case_index,
                error_type=type(exc).__name__,
                error_message=str(exc),
                device_id=device_id,
            )

        case_result = payload["case_result"]
        case_result["device_id"] = device_id
        completed_record = None
        completed_log = None
        with self._condition:
            state = self._op_states[work_item.op_index]
            state.result["cases"].append(case_result)
            state.completed_cases += 1
            state.devices_used.add(device_id)
            state.result["correctness"]["failed_trials"] = [
                failure
                for case in state.result["cases"]
                for failure in (case.get("failed_trials") or [])
            ]
            running_cases = self._running_cases_for_state_locked(state.index)
            if state.completed_cases >= state.total_cases:
                finalize_generated_operator_result(
                    result=state.result,
                    shape_mode=state.task.shape_mode,
                    random_correctness_config=state.random_correctness_config,
                    total_cases=state.total_cases,
                )
                state.result["devices"] = sorted(state.devices_used) or list(self.args.device_ids)
                write_partial_generated_operator_result(
                    partial_result_path=self._partial_path(state),
                    result=state.result,
                    status="done",
                    partial_result_metadata=state.partial_metadata,
                    total_cases=state.total_cases,
                    completed_cases=state.completed_cases,
                    last_completed_case_index=work_item.case_index,
                    running_cases=[],
                )
                _cleanup_torch_extensions(state.op_run_dir)
                completed_record = _batch_record_for_task(
                    state.task,
                    None,
                    state.op_run_dir,
                    state.result,
                )
                completed_log = {
                    "op_name": state.task.op_name,
                    "success": evaluation_success(state.result),
                    "elapsed": time.monotonic() - (state.first_case_started_at or state.started_at),
                }
                self._records[state.index] = completed_record
                self._op_states.pop(state.index, None)
            else:
                write_partial_generated_operator_result(
                    partial_result_path=self._partial_path(state),
                    result=state.result,
                    status="running",
                    partial_result_metadata=state.partial_metadata,
                    total_cases=state.total_cases,
                    completed_cases=state.completed_cases,
                    last_completed_case_index=work_item.case_index,
                    running_cases=running_cases,
                )
            self._schedule_cases_locked()
            self._condition.notify_all()

        if completed_log is not None and self.log_fn is not None:
            self.log_fn(
                "eval",
                "done",
                op_name=completed_log["op_name"],
                device_id=device_id,
                success=completed_log["success"],
                elapsed=completed_log["elapsed"],
            )

    def finish(self, *, write_summary: bool = True) -> dict:
        try:
            with self._condition:
                while any(record is None for record in self._records):
                    self._condition.wait(timeout=0.1)
                finalized = [record for record in self._records if record is not None]
        finally:
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=1.0)
            self._prepare_executor.shutdown(wait=True)
            for executor in self._executors.values():
                executor.shutdown(wait=True)

        payload = {
            **_summary_payload(
                records=finalized,
                summary_context=self.summary_context,
                run_dir=self.run_dir,
                device_ids=self.args.device_ids,
            ),
        }
        if write_summary:
            _write_passk_op_summaries(finalized)
            _write_category_summaries(
                run_dir=self.run_dir,
                records=finalized,
                summary_context=self.summary_context,
                device_ids=self.args.device_ids,
            )
            write_json(self.run_dir / "evaluation.json", payload)
        return payload


StreamingEvaluationScheduler = CaseLevelStreamingEvaluationScheduler


def run_single_evaluation_task(
    *,
    args: Any,
    run_dir: Path,
    task: EvaluationTask,
    log_fn: LogFn | None = None,
) -> dict:
    scheduler = CaseLevelStreamingEvaluationScheduler(
        args=args,
        run_dir=run_dir,
        summary_context={"run_id": run_dir.name, "selector": {"type": "op", "value": task.op_name}},
        log_fn=log_fn,
    )
    scheduler.submit(task, device_id=args.device_ids[0], op_run_dir=run_dir)
    scheduler.finish(write_summary=False)
    return json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))


def run_evaluation_tasks(
    *,
    args: Any,
    run_dir: Path,
    tasks: list[EvaluationTask],
    summary_context: dict[str, Any],
    log_fn: LogFn | None = None,
) -> dict:
    scheduler = StreamingEvaluationScheduler(
        args=args,
        run_dir=run_dir,
        summary_context=summary_context,
        log_fn=log_fn,
    )
    for index, task in enumerate(tasks):
        scheduler.submit(
            task,
            device_id=args.device_ids[index % len(args.device_ids)],
            op_run_dir=_task_run_dir(run_dir, task),
        )
    return scheduler.finish()
