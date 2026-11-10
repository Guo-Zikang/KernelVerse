"""Task discovery helpers for generated-code evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from operators.registry import get_operator_spec, resolve_operator_selection
from utils.run_paths import (
    existing_op_run_dir,
    legacy_op_run_dir,
    op_run_dir,
)


@dataclass
class EvaluationTask:
    op_name: str
    shape_mode: str
    shape_name: str
    shape_family: str
    language: str
    generated_path: Path
    random_count: int | None = None
    random_max_attempts: int | None = None
    candidate_id: int | None = None
    samples_per_op: int = 1


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_reeval_run_id(source_run_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{source_run_id}_reeval_{timestamp}"


def default_compile_run_id(source_run_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{source_run_id}_compile_{timestamp}"


def source_run_dir(run_id: str) -> Path:
    return repo_root() / "runs" / run_id


def batch_generated_path(run_dir: Path, generated_dir: str | None, op_name: str) -> Path:
    if generated_dir:
        return Path(generated_dir).resolve() / f"{op_name}.txt"
    return existing_op_run_dir(run_dir, op_name) / "generated_ops" / f"{op_name}.txt"


def normal_task(args: Any, run_dir: Path, op_name: str) -> EvaluationTask:
    return EvaluationTask(
        op_name=op_name,
        shape_mode=args.shape_mode,
        shape_name=args.shape_name,
        shape_family=args.shape_family,
        language=args.language,
        generated_path=batch_generated_path(run_dir, args.generated_dir, op_name),
        random_count=args.random_count,
        random_max_attempts=args.random_max_attempts,
    )


def single_normal_task(args: Any, generated_path: Path, op_name: str) -> EvaluationTask:
    return EvaluationTask(
        op_name=op_name,
        shape_mode=args.shape_mode,
        shape_name=args.shape_name,
        shape_family=args.shape_family,
        language=args.language,
        generated_path=generated_path,
        random_count=args.random_count,
        random_max_attempts=args.random_max_attempts,
    )


def _generation_metadata(source_op_dir: Path, batch_metadata: dict[str, Any]) -> dict[str, Any]:
    generation_path = source_op_dir / "generation.json"
    if generation_path.exists():
        return read_json(generation_path)
    return {
        "shape_mode": batch_metadata.get("shape_mode"),
        "shape_name": batch_metadata.get("shape_name"),
        "shape_family": batch_metadata.get("shape_family"),
        "language": batch_metadata.get("language"),
    }


def _task_from_generation(
    *,
    op_name: str,
    generated_path: Path,
    generation: dict[str, Any],
    batch_metadata: dict[str, Any],
    candidate_id: int | None = None,
    samples_per_op: int | None = None,
) -> EvaluationTask:
    random_shape = generation.get("random_shape") or {}
    shape_mode = generation.get("shape_mode") or batch_metadata.get("shape_mode")
    language = generation.get("language") or batch_metadata.get("language") or "cuda"
    if not shape_mode:
        raise ValueError(f"Missing shape_mode metadata for generated operator: {op_name}")
    return EvaluationTask(
        op_name=op_name,
        shape_mode=shape_mode,
        shape_name=generation.get("shape_name") or batch_metadata.get("shape_name") or "default",
        shape_family=generation.get("shape_family") or batch_metadata.get("shape_family") or "all",
        language=language,
        generated_path=generated_path,
        random_count=random_shape.get("count"),
        random_max_attempts=random_shape.get("max_attempts"),
        candidate_id=(
            candidate_id
            if candidate_id is not None
            else generation.get("candidate_id")
        ),
        samples_per_op=int(
            samples_per_op
            or generation.get("samples_per_op")
            or batch_metadata.get("samples_per_op")
            or 1
        ),
    )


def task_from_generation(
    *,
    op_name: str,
    generated_path: Path,
    generation: dict[str, Any],
    batch_metadata: dict[str, Any],
    candidate_id: int | None = None,
    samples_per_op: int | None = None,
) -> EvaluationTask:
    """Build one evaluation task from generation metadata."""
    return _task_from_generation(
        op_name=op_name,
        generated_path=generated_path,
        generation=generation,
        batch_metadata=batch_metadata,
        candidate_id=candidate_id,
        samples_per_op=samples_per_op,
    )


def _candidate_id_from_record(record: dict[str, Any]) -> int | None:
    value = record.get("candidate_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_source_tasks(source_dir: Path) -> tuple[list[EvaluationTask], dict[str, Any]]:
    batch_path = source_dir / "batch.json"
    if batch_path.exists():
        batch_metadata = read_json(batch_path)
        records_by_op = {
            str(record["op_name"]): record
            for record in batch_metadata.get("ops", [])
            if record.get("op_name")
        }
        op_names = list(records_by_op)
        if not op_names:
            ops_dir = source_dir / "ops"
            if ops_dir.exists():
                grouped_op_names = {
                    path.name
                    for path in ops_dir.glob("*/*")
                    if path.is_dir() and (path / "generation.json").exists()
                }
                legacy_op_names = {
                    path.name
                    for path in ops_dir.iterdir()
                    if path.is_dir() and (path / "generation.json").exists()
                }
                op_names = sorted(grouped_op_names | legacy_op_names)
        tasks = []
        for op_name in op_names:
            record = records_by_op.get(op_name) or {}
            source_op_dir = (
                Path(record["run_dir"])
                if record.get("run_dir")
                else existing_op_run_dir(source_dir, op_name)
            )
            candidate_records = [
                candidate
                for candidate in (record.get("candidates") or [])
                if isinstance(candidate, dict)
            ]
            if candidate_records:
                for candidate in sorted(
                    candidate_records,
                    key=lambda item: int(item.get("candidate_id", 0)),
                ):
                    candidate_dir = (
                        Path(candidate["run_dir"])
                        if candidate.get("run_dir")
                        else source_op_dir / "candidates" / f"candidate_{int(candidate.get('candidate_id', 0)):02d}"
                    )
                    generated_path = (
                        Path(candidate["generated_path"])
                        if candidate.get("generated_path")
                        else candidate_dir / "generated_ops" / f"{op_name}.txt"
                    )
                    generation = _generation_metadata(candidate_dir, batch_metadata)
                    tasks.append(
                        task_from_generation(
                            op_name=op_name,
                            generated_path=generated_path,
                            generation=generation,
                            batch_metadata=batch_metadata,
                            candidate_id=_candidate_id_from_record(candidate),
                            samples_per_op=record.get("samples_per_op"),
                        )
                    )
                continue

            candidates_dir = source_op_dir / "candidates"
            if candidates_dir.exists():
                discovered_candidates = sorted(
                    path
                    for path in candidates_dir.glob("candidate_*")
                    if path.is_dir() and (path / "generation.json").exists()
                )
                if discovered_candidates:
                    for candidate_dir in discovered_candidates:
                        candidate_name = candidate_dir.name.removeprefix("candidate_")
                        try:
                            candidate_id = int(candidate_name)
                        except ValueError:
                            candidate_id = None
                        generated_path = candidate_dir / "generated_ops" / f"{op_name}.txt"
                        generation = _generation_metadata(candidate_dir, batch_metadata)
                        tasks.append(
                            task_from_generation(
                                op_name=op_name,
                                generated_path=generated_path,
                                generation=generation,
                                batch_metadata=batch_metadata,
                                candidate_id=candidate_id,
                                samples_per_op=record.get("samples_per_op"),
                            )
                        )
                    continue

            generated_path = (
                Path(record["generated_path"])
                if record.get("generated_path")
                else source_op_dir / "generated_ops" / f"{op_name}.txt"
            )
            generation = _generation_metadata(source_op_dir, batch_metadata)
            tasks.append(
                task_from_generation(
                    op_name=op_name,
                    generated_path=generated_path,
                    generation=generation,
                    batch_metadata=batch_metadata,
                )
            )
        selector = batch_metadata.get("selector") or {"type": "all", "value": "all"}
        return tasks, selector

    generation_path = source_dir / "generation.json"
    if not generation_path.exists():
        raise ValueError(f"Run directory does not contain batch.json or generation.json: {source_dir}")
    generation = read_json(generation_path)
    op_name = generation.get("op_name")
    if not op_name:
        raise ValueError(f"Missing op_name in generation metadata: {generation_path}")
    generated_path = source_dir / "generated_ops" / f"{op_name}.txt"
    task = task_from_generation(
        op_name=op_name,
        generated_path=generated_path,
        generation=generation,
        batch_metadata={},
    )
    return [task], {"type": "op", "value": op_name}


def select_source_tasks(
    *,
    args: Any,
    source_tasks: list[EvaluationTask],
    source_selector: dict[str, Any],
) -> tuple[list[EvaluationTask], dict[str, Any]]:
    if args.op:
        requested_ops = resolve_operator_selection(op_name=args.op)
        requested_set = set(requested_ops)
        selector = (
            {"type": "op", "value": requested_ops[0]}
            if len(requested_ops) == 1
            else {"type": "ops", "value": requested_ops}
        )
        selected = [task for task in source_tasks if task.op_name in requested_set]
        discovered_ops = {task.op_name for task in selected}
        missing_ops = [op_name for op_name in requested_ops if op_name not in discovered_ops]
        if missing_ops:
            raise ValueError(
                "Generated code was not found in the source run for operator(s): "
                + ", ".join(missing_ops)
            )
    elif args.category:
        allowed_ops = set(resolve_operator_selection(category=args.category))
        selector = {"type": "category", "value": args.category}
        selected = [task for task in source_tasks if task.op_name in allowed_ops]
    elif args.all:
        selector = {"type": "all", "value": "all"}
        selected = list(source_tasks)
    else:
        selector_type = str(source_selector.get("type") or "all")
        selector_value = source_selector.get("value") or "all"
        selector = {"type": selector_type, "value": selector_value}
        if selector_type == "op":
            selected = [task for task in source_tasks if task.op_name == str(selector_value)]
        elif selector_type == "ops":
            selected_ops = (
                {str(op_name) for op_name in selector_value}
                if isinstance(selector_value, list)
                else set()
            )
            selected = [task for task in source_tasks if task.op_name in selected_ops]
        elif selector_type == "category":
            selected = [
                task for task in source_tasks
                if get_operator_spec(task.op_name).get("category") == str(selector_value)
            ]
        else:
            selector = {"type": "all", "value": "all"}
            selected = list(source_tasks)

    if not selected:
        raise ValueError(
            f"No generated operators matched selector {selector['type']}={selector['value']} "
            "in the source run."
        )
    return selected, selector


def common_task_value(tasks: list[EvaluationTask], field_name: str) -> Any:
    values = {getattr(task, field_name) for task in tasks}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"
