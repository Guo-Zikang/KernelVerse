"""Compile-only scheduling and summaries for generated operator code."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable

from evaluator.tasks import EvaluationTask
from operators.registry import get_operator_spec
from runner.generated_code_checker import check_generated_file
from runner.generated_correctness_runner import compile_generated_operator
from utils.run_paths import op_run_dir as category_op_run_dir


LogFn = Callable[..., None]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_dir(op_run_dir: Path, candidate_id: int) -> Path:
    return op_run_dir / "candidates" / f"candidate_{candidate_id:02d}"


def task_run_dir(run_dir: Path, task: EvaluationTask) -> Path:
    op_run_dir = category_op_run_dir(run_dir, task.op_name)
    if task.candidate_id is None:
        return op_run_dir
    return _candidate_dir(op_run_dir, int(task.candidate_id))


def _torch_extensions_dir(op_run_dir: Path, device_id: int) -> Path:
    return op_run_dir / "torch_extensions" / f"device_{device_id}"


def _cleanup_torch_extensions(op_run_dir: Path) -> None:
    shutil.rmtree(op_run_dir / "torch_extensions", ignore_errors=True)


def compilation_success(result: dict[str, Any]) -> bool:
    return bool(result.get("success"))


def _resume_result(op_run_dir: Path, task: EvaluationTask) -> dict[str, Any] | None:
    if not (op_run_dir / "compilation.json").exists():
        return None
    try:
        result = json.loads((op_run_dir / "compilation.json").read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-except
        return None
    if result.get("op_name") != task.op_name:
        return None
    try:
        generated_matches = Path(str(result.get("generated_code"))).resolve() == task.generated_path.resolve()
    except Exception:
        generated_matches = False
    if not generated_matches or "compiled" not in result or "success" not in result:
        return None
    return result


def _compilation_record(
    task: EvaluationTask,
    *,
    device_id: int,
    op_run_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "op_name": task.op_name,
        "category": get_operator_spec(task.op_name)["category"],
        "device_id": device_id,
        "run_dir": str(op_run_dir),
        "compilation_json": str(op_run_dir / "compilation.json"),
        "generated_path": str(task.generated_path),
        "compiled": bool(result.get("compiled")),
        "model_new_found": bool(result.get("model_new_found")),
        "success": compilation_success(result),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
    }
    reward_hacking = result.get("reward_hacking") or {}
    record["reward_hacking_risk_level"] = reward_hacking.get("risk_level")
    if task.candidate_id is not None:
        record["candidate_id"] = int(task.candidate_id)
        record["candidate_name"] = f"candidate_{int(task.candidate_id):02d}"
        record["samples_per_op"] = int(task.samples_per_op)
    return record


def _failure_result(
    task: EvaluationTask,
    *,
    device_id: int,
    run_id: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    result = {
        "op_name": task.op_name,
        "language": task.language,
        "device_id": device_id,
        "device": None,
        "run_id": run_id,
        "generated_code": str(task.generated_path),
        "reward_hacking": {"risk_level": "low"},
        "compiled": False,
        "model_new_found": False,
        "success": False,
        "elapsed_seconds": 0.0,
        "compile_info": error_message,
        "logs": {},
        "error_type": error_type,
        "error_message": error_message,
    }
    if task.candidate_id is not None:
        result["candidate_id"] = int(task.candidate_id)
        result["samples_per_op"] = int(task.samples_per_op)
    return result


def run_compilation_task(
    *,
    task: EvaluationTask,
    device_id: int,
    op_run_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Compile one generated candidate and persist its detailed result."""
    op_run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = op_run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts_dir / "compile_stdout.txt"
    stderr_path = artifacts_dir / "compile_stderr.txt"
    reward_report_path = op_run_dir / "reward_hacking_report.json"

    try:
        reward_report = check_generated_file(task.generated_path, op_name=task.op_name)
    except Exception as exc:  # pylint: disable=broad-except
        reward_report = {"risk_level": "low", "checker_error": str(exc)}
    _write_json(reward_report_path, reward_report)

    _cleanup_torch_extensions(op_run_dir)
    started_at = time.monotonic()
    try:
        compile_result = compile_generated_operator(
            op_name=task.op_name,
            language=task.language,
            generated_path=str(task.generated_path),
            device_id=device_id,
            torch_extensions_dir=str(_torch_extensions_dir(op_run_dir, device_id)),
        )
    except Exception as exc:  # pylint: disable=broad-except
        compile_result = {
            "compiled": False,
            "model_new_found": False,
            "success": False,
            "stdout": "",
            "stderr": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "compile_info": str(exc),
            "device": None,
        }
    elapsed = time.monotonic() - started_at

    stdout_path.write_text(str(compile_result.pop("stdout", "") or ""), encoding="utf-8")
    stderr_path.write_text(str(compile_result.pop("stderr", "") or ""), encoding="utf-8")
    result = {
        "op_name": task.op_name,
        "language": task.language,
        "device_id": device_id,
        "device": compile_result.get("device"),
        "run_id": run_id,
        "generated_code": str(task.generated_path),
        "reward_hacking": {"risk_level": reward_report.get("risk_level", "low")},
        "compiled": bool(compile_result.get("compiled")),
        "model_new_found": bool(compile_result.get("model_new_found")),
        "success": bool(compile_result.get("success")),
        "elapsed_seconds": round(elapsed, 4),
        "compile_info": compile_result.get("compile_info"),
        "logs": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "error_type": compile_result.get("error_type"),
        "error_message": compile_result.get("error_message"),
    }
    if task.candidate_id is not None:
        result["candidate_id"] = int(task.candidate_id)
        result["samples_per_op"] = int(task.samples_per_op)
    _write_json(op_run_dir / "compilation.json", result)
    _cleanup_torch_extensions(op_run_dir)
    return result


def _op_dir_from_candidate_record(record: dict[str, Any]) -> Path:
    candidate_dir = Path(str(record["run_dir"]))
    return candidate_dir.parent.parent


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    summary_context: dict[str, Any],
    run_dir: Path,
    device_ids: list[int],
    category: str | None = None,
) -> dict[str, Any]:
    candidate_mode = any(record.get("candidate_id") is not None for record in records)
    if candidate_mode:
        records_by_op: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            records_by_op.setdefault(str(record["op_name"]), []).append(record)
        op_records = []
        for op_name, candidates in sorted(records_by_op.items()):
            candidates.sort(key=lambda item: int(item.get("candidate_id", 0)))
            compiled_candidates = sum(1 for item in candidates if item.get("success"))
            first = candidates[0]
            op_dir = _op_dir_from_candidate_record(first)
            op_record = {
                "op_name": op_name,
                "category": first["category"],
                "device_ids": sorted({int(item["device_id"]) for item in candidates}),
                "run_dir": str(op_dir),
                "compilation_json": str(op_dir / "compilation.json"),
                "samples_per_op": max(int(item.get("samples_per_op") or 1) for item in candidates),
                "total_candidates": len(candidates),
                "compiled_candidates": compiled_candidates,
                "failed_candidates": len(candidates) - compiled_candidates,
                "success": compiled_candidates == len(candidates),
                "candidates": candidates,
            }
            if not op_record["success"]:
                op_record["error_type"] = "CandidateCompilationFailed"
                op_record["error_message"] = "One or more generated candidates failed compilation."
            _write_json(op_dir / "compilation.json", op_record)
            op_records.append(op_record)
        total_candidates = sum(item["total_candidates"] for item in op_records)
        compiled_candidates = sum(item["compiled_candidates"] for item in op_records)
        payload = {
            **summary_context,
            "run_dir": str(run_dir),
            "device_ids": device_ids,
            "total_ops": len(op_records),
            "successful_ops": sum(1 for item in op_records if item["success"]),
            "failed_ops": sum(1 for item in op_records if not item["success"]),
            "total_candidates": total_candidates,
            "compiled_candidates": compiled_candidates,
            "failed_candidates": total_candidates - compiled_candidates,
            "success": all(item["success"] for item in op_records),
            "ops": op_records,
        }
    else:
        payload = {
            **summary_context,
            "run_dir": str(run_dir),
            "device_ids": device_ids,
            "total_ops": len(records),
            "successful_ops": sum(1 for item in records if item["success"]),
            "failed_ops": sum(1 for item in records if not item["success"]),
            "success": all(item["success"] for item in records),
            "ops": records,
        }
    if category is not None:
        payload["category"] = category
    return payload


def _write_category_summaries(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    summary_context: dict[str, Any],
    device_ids: list[int],
) -> None:
    records_by_category: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_category.setdefault(str(record["category"]), []).append(record)
    for category, category_records in records_by_category.items():
        category_dir = run_dir / "ops" / category
        payload = _summarize_records(
            category_records,
            summary_context=summary_context,
            run_dir=category_dir,
            device_ids=device_ids,
            category=category,
        )
        _write_json(category_dir / "compilation.json", payload)


class StreamingCompilationScheduler:
    """Compile generated candidates as soon as generation completes."""

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
        self._records: list[dict[str, Any] | None] = []
        self._futures: dict[Future, tuple[int, EvaluationTask, int, Path, float]] = {}
        self._condition = threading.Condition()
        self._executors = {
            device_id: ThreadPoolExecutor(max_workers=1)
            for device_id in args.device_ids
        }

    def submit(self, task: EvaluationTask, *, device_id: int, op_run_dir: Path) -> None:
        with self._condition:
            index = len(self._records)
            self._records.append(None)
        if self.log_fn is not None:
            self.log_fn(
                "compile",
                "queued",
                op_name=task.op_name,
                device_id=device_id,
                message=(f"candidate={task.candidate_id}" if task.candidate_id is not None else None),
            )
        if bool(getattr(self.args, "resume", False)):
            resumed_result = _resume_result(op_run_dir, task)
            if resumed_result is not None:
                record = _compilation_record(
                    task,
                    device_id=int(resumed_result.get("device_id", device_id)),
                    op_run_dir=op_run_dir,
                    result=resumed_result,
                )
                with self._condition:
                    self._records[index] = record
                    self._condition.notify_all()
                if self.log_fn is not None:
                    self.log_fn(
                        "compile",
                        "done",
                        op_name=task.op_name,
                        device_id=record["device_id"],
                        success=record["success"],
                        elapsed=0.0,
                        message="resumed=true",
                    )
                return
        future = self._executors[device_id].submit(
            run_compilation_task,
            task=task,
            device_id=device_id,
            op_run_dir=op_run_dir,
            run_id=f"{self.run_dir.name}_{task.op_name}",
        )
        with self._condition:
            self._futures[future] = (index, task, device_id, op_run_dir, time.monotonic())
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
        result = _failure_result(
            task,
            device_id=device_id,
            run_id=f"{self.run_dir.name}_{task.op_name}",
            error_type=error_type or type(error).__name__,
            error_message=str(error),
        )
        op_run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(op_run_dir / "compilation.json", result)
        record = _compilation_record(task, device_id=device_id, op_run_dir=op_run_dir, result=result)
        with self._condition:
            self._records.append(record)
            self._condition.notify_all()
        if self.log_fn is not None:
            self.log_fn("compile", "queued", op_name=task.op_name, device_id=device_id)
            self.log_fn("compile", "done", op_name=task.op_name, device_id=device_id, success=False, elapsed=0.0)

    def _complete_future(self, future: Future) -> None:
        with self._condition:
            metadata = self._futures.pop(future, None)
        if metadata is None:
            return
        index, task, device_id, op_run_dir, started_at = metadata
        try:
            try:
                result = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                result = _failure_result(
                    task,
                    device_id=device_id,
                    run_id=f"{self.run_dir.name}_{task.op_name}",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                _write_json(op_run_dir / "compilation.json", result)
            record = _compilation_record(
                task,
                device_id=device_id,
                op_run_dir=op_run_dir,
                result=result,
            )
        except Exception as exc:  # pylint: disable=broad-except
            record = {
                "op_name": task.op_name,
                "category": get_operator_spec(task.op_name)["category"],
                "device_id": device_id,
                "run_dir": str(op_run_dir),
                "compilation_json": str(op_run_dir / "compilation.json"),
                "generated_path": str(task.generated_path),
                "compiled": False,
                "model_new_found": False,
                "success": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        finally:
            with self._condition:
                self._records[index] = record
                self._condition.notify_all()
        if self.log_fn is not None:
            self.log_fn(
                "compile",
                "done",
                op_name=task.op_name,
                device_id=device_id,
                success=record["success"],
                elapsed=time.monotonic() - started_at,
                message=(f"candidate={task.candidate_id}" if task.candidate_id is not None else None),
            )

    def finish(self, *, write_summary: bool = True) -> dict[str, Any]:
        try:
            with self._condition:
                while any(record is None for record in self._records):
                    self._condition.wait(timeout=0.1)
                records = [record for record in self._records if record is not None]
        finally:
            for executor in self._executors.values():
                executor.shutdown(wait=True)
        payload = _summarize_records(
            records,
            summary_context=self.summary_context,
            run_dir=self.run_dir,
            device_ids=list(self.args.device_ids),
        )
        if write_summary:
            _write_category_summaries(
                run_dir=self.run_dir,
                records=records,
                summary_context=self.summary_context,
                device_ids=list(self.args.device_ids),
            )
            _write_json(self.run_dir / "compilation.json", payload)
        return payload


def run_single_compilation_task(
    *,
    args: Any,
    run_dir: Path,
    task: EvaluationTask,
    log_fn: LogFn | None = None,
) -> dict[str, Any]:
    scheduler = StreamingCompilationScheduler(
        args=args,
        run_dir=run_dir,
        summary_context={"run_id": run_dir.name, "selector": {"type": "op", "value": task.op_name}},
        log_fn=log_fn,
    )
    scheduler.submit(task, device_id=args.device_ids[0], op_run_dir=run_dir)
    scheduler.finish(write_summary=False)
    return json.loads((run_dir / "compilation.json").read_text(encoding="utf-8"))


def run_compilation_tasks(
    *,
    args: Any,
    run_dir: Path,
    tasks: list[EvaluationTask],
    summary_context: dict[str, Any],
    log_fn: LogFn | None = None,
) -> dict[str, Any]:
    scheduler = StreamingCompilationScheduler(
        args=args,
        run_dir=run_dir,
        summary_context=summary_context,
        log_fn=log_fn,
    )
    for index, task in enumerate(tasks):
        scheduler.submit(
            task,
            device_id=args.device_ids[index % len(args.device_ids)],
            op_run_dir=task_run_dir(run_dir, task),
        )
    return scheduler.finish()
