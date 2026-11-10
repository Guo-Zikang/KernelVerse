"""Batch scheduling for generated-code creation tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from codegen.artifacts import write_json
from codegen.generators import GenerationRequest, run_generation_with_retries
from operators.registry import get_operator_spec
from utils.run_paths import op_run_dir as category_op_run_dir


LogFn = Callable[..., None]
GenerationDoneFn = Callable[[dict, dict], None]
PASSK_TEMPERATURE = 0.6


@dataclass
class GenerationBatchRequest:
    run_id: str
    run_dir: Path
    op_names: list[str]
    selector: dict
    shape_mode: str
    shape_name: str
    shape_family: str
    language: str
    device_ids: list[int]
    llm_workers: int = 200
    env_path: Path | None = None
    no_overwrite: bool = False
    random_count: int | None = None
    random_max_attempts: int | None = None
    samples_per_op: int = 1
    generation_retries: int = 5


def _shape_context(request: GenerationBatchRequest) -> dict:
    context = {
        "shape_mode": request.shape_mode,
    }
    if request.shape_mode == "single":
        context["shape_name"] = request.shape_name
    elif request.shape_mode == "family":
        context["shape_family"] = request.shape_family
    return context


def _reward_hacking_risk_level(generation: dict) -> str | None:
    reward_hacking = generation.get("reward_hacking") or {}
    return reward_hacking.get("risk_level")


def _generation_record(
    *,
    op_name: str,
    category: str,
    device_id: int,
    op_run_dir: Path,
    generated_path: Path,
    generation: dict,
) -> dict:
    generation_success = bool(generation.get("success"))
    record = {
        "op_name": op_name,
        "category": category,
        "device_id": device_id,
        "run_dir": str(op_run_dir),
        "generation_json": str(op_run_dir / "generation.json"),
        "generated_path": str(generated_path),
        "reward_hacking_risk_level": _reward_hacking_risk_level(generation),
        "generation_success": generation_success,
        "success": generation_success,
        "error_type": generation.get("error_type") if not generation_success else None,
        "error_message": generation.get("error_message") if not generation_success else None,
    }
    candidate_id = generation.get("candidate_id")
    if candidate_id is not None:
        record["candidate_id"] = int(candidate_id)
        record["candidate_name"] = f"candidate_{int(candidate_id):02d}"
    return record


def _generation_failure(
    *,
    request: GenerationBatchRequest,
    op_run_id: str,
    op_name: str,
    op_run_dir: Path,
    error: Exception,
    candidate_id: int | None = None,
    samples_per_op: int = 1,
) -> dict:
    generation = {
        "run_id": op_run_id,
        "op_name": op_name,
        "shape_mode": request.shape_mode,
        "language": request.language,
        "generator": {
            "type": "llm",
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        "success": False,
    }
    if candidate_id is not None:
        generation["candidate_id"] = candidate_id
        generation["samples_per_op"] = samples_per_op
    if request.shape_mode == "single":
        generation["shape_name"] = request.shape_name
    elif request.shape_mode == "family":
        generation["shape_family"] = request.shape_family
    write_json(op_run_dir / "generation.json", generation)
    return generation


def _generate_one(
    *,
    request: GenerationBatchRequest,
    op_name: str,
    device_id: int,
    op_run_id: str,
    op_run_dir: Path,
    log_fn: LogFn | None,
    candidate_id: int | None = None,
    samples_per_op: int = 1,
) -> dict:
    start = time.monotonic()
    if log_fn is not None:
        log_fn(
            "codegen",
            "start",
            op_name=op_name,
            device_id=device_id,
            message=(
                f"candidate={candidate_id}"
                if candidate_id is not None
                else None
            ),
        )
    try:
        generation = run_generation_with_retries(
            GenerationRequest(
                run_id=op_run_id,
                op_name=op_name,
                shape_mode=request.shape_mode,
                shape_name=request.shape_name,
                shape_family=request.shape_family,
                language=request.language,
                run_dir=op_run_dir,
                device_id=device_id,
                env_path=request.env_path,
                no_overwrite=request.no_overwrite,
                random_count=request.random_count,
                random_max_attempts=request.random_max_attempts,
                candidate_id=candidate_id,
                samples_per_op=samples_per_op,
                temperature=PASSK_TEMPERATURE if samples_per_op > 1 else None,
            ),
            generation_retries=request.generation_retries,
            on_retry=(
                lambda retry, total, _result: log_fn(
                    "codegen",
                    "retry",
                    op_name=op_name,
                    device_id=device_id,
                    success=False,
                    message=(
                        f"retry={retry}/{total}"
                        + (
                            f" candidate={candidate_id}"
                            if candidate_id is not None
                            else ""
                        )
                    ),
                )
                if log_fn is not None
                else None
            ),
        )
    except Exception:
        if log_fn is not None:
            log_fn(
                "codegen",
                "done",
                op_name=op_name,
                device_id=device_id,
                success=False,
                elapsed=time.monotonic() - start,
                message=(
                    f"candidate={candidate_id}"
                    if candidate_id is not None
                    else None
                ),
            )
        raise
    if log_fn is not None:
        log_fn(
            "codegen",
            "done",
            op_name=op_name,
            device_id=device_id,
            success=bool(generation.get("success")),
            elapsed=time.monotonic() - start,
            message=(
                f"candidate={candidate_id}"
                if candidate_id is not None
                else None
            ),
        )
    return generation


def _batch_payload(
    *,
    request: GenerationBatchRequest,
    records: list[dict],
) -> dict:
    successful_ops = sum(1 for record in records if record.get("generation_success"))
    payload = {
        "success": successful_ops == len(records),
        "run_id": request.run_id,
        "run_dir": str(request.run_dir),
        "selector": request.selector,
        **_shape_context(request),
        "language": request.language,
        "device_ids": request.device_ids,
        "llm_workers": request.llm_workers,
        "generation_retries": request.generation_retries,
        "samples_per_op": request.samples_per_op,
        "total_ops": len(records),
        "successful_ops": successful_ops,
        "failed_ops": len(records) - successful_ops,
        "ops": records,
    }
    return payload


def _candidate_dir(op_run_dir: Path, candidate_id: int) -> Path:
    return op_run_dir / "candidates" / f"candidate_{candidate_id:02d}"


def _candidate_generation_record(
    *,
    op_name: str,
    category: str,
    device_id: int,
    op_run_dir: Path,
    candidate_id: int,
    generation: dict,
) -> dict:
    candidate_run_dir = _candidate_dir(op_run_dir, candidate_id)
    generated_path = candidate_run_dir / "generated_ops" / f"{op_name}.txt"
    return _generation_record(
        op_name=op_name,
        category=category,
        device_id=device_id,
        op_run_dir=candidate_run_dir,
        generated_path=generated_path,
        generation=generation,
    )


def _op_passk_generation_summary(
    *,
    request: GenerationBatchRequest,
    op_name: str,
    category: str,
    op_run_id: str,
    op_run_dir: Path,
    candidates: list[dict],
) -> dict:
    successful_candidates = sum(1 for item in candidates if item.get("generation_success"))
    summary = {
        "run_id": op_run_id,
        "op_name": op_name,
        "category": category,
        "shape_mode": request.shape_mode,
        "language": request.language,
        "samples_per_op": request.samples_per_op,
        "temperature": PASSK_TEMPERATURE,
        "generation_retries": request.generation_retries,
        "generation_success": successful_candidates > 0,
        "success": successful_candidates > 0,
        "total_candidates": len(candidates),
        "successful_candidates": successful_candidates,
        "failed_candidates": len(candidates) - successful_candidates,
        "candidates": candidates,
    }
    if request.shape_mode == "single":
        summary["shape_name"] = request.shape_name
    elif request.shape_mode == "family":
        summary["shape_family"] = request.shape_family
    write_json(op_run_dir / "generation.json", summary)
    return summary


def _op_passk_record(
    *,
    op_name: str,
    category: str,
    device_id: int,
    op_run_dir: Path,
    summary: dict,
) -> dict:
    generation_success = bool(summary.get("generation_success"))
    return {
        "op_name": op_name,
        "category": category,
        "device_id": device_id,
        "run_dir": str(op_run_dir),
        "generation_json": str(op_run_dir / "generation.json"),
        "samples_per_op": summary.get("samples_per_op"),
        "total_candidates": summary.get("total_candidates"),
        "successful_candidates": summary.get("successful_candidates"),
        "failed_candidates": summary.get("failed_candidates"),
        "candidates": summary.get("candidates") or [],
        "generation_success": generation_success,
        "success": generation_success,
        "error_type": None if generation_success else "AllCandidatesFailed",
        "error_message": None if generation_success else "All pass@k candidates failed generation.",
    }


def run_generation_batch(
    request: GenerationBatchRequest,
    *,
    log_fn: LogFn | None = None,
    on_generation_done: GenerationDoneFn | None = None,
) -> dict:
    """Run code generation for multiple operators and write batch.json."""
    if request.samples_per_op > 1:
        return _run_generation_batch_passk(
            request,
            log_fn=log_fn,
            on_generation_done=on_generation_done,
        )

    records: list[dict | None] = [None] * len(request.op_names)
    max_workers = max(1, request.llm_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, op_name in enumerate(request.op_names):
            device_id = request.device_ids[index % len(request.device_ids)]
            op_run_id = f"{request.run_id}_{op_name}"
            op_run_dir = category_op_run_dir(request.run_dir, op_name)
            future = executor.submit(
                _generate_one,
                request=request,
                op_name=op_name,
                device_id=device_id,
                op_run_id=op_run_id,
                op_run_dir=op_run_dir,
                log_fn=log_fn,
            )
            futures[future] = (index, op_name, device_id, op_run_id, op_run_dir)

        for future in as_completed(futures):
            index, op_name, device_id, op_run_id, op_run_dir = futures[future]
            category = get_operator_spec(op_name)["category"]
            generated_path = op_run_dir / "generated_ops" / f"{op_name}.txt"
            try:
                generation = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                generation = _generation_failure(
                    request=request,
                    op_run_id=op_run_id,
                    op_name=op_name,
                    op_run_dir=op_run_dir,
                    error=exc,
                )
            record = _generation_record(
                op_name=op_name,
                category=category,
                device_id=device_id,
                op_run_dir=op_run_dir,
                generated_path=generated_path,
                generation=generation,
            )
            records[index] = record
            if on_generation_done is not None:
                on_generation_done(record, generation)

    finalized = [
        record
        if record is not None
        else {
            "op_name": request.op_names[index],
            "category": get_operator_spec(request.op_names[index])["category"],
            "device_id": request.device_ids[index % len(request.device_ids)],
            "generation_success": False,
            "success": False,
            "error_message": "Operator did not produce a generation record.",
        }
        for index, record in enumerate(records)
    ]
    payload = _batch_payload(request=request, records=finalized)
    write_json(request.run_dir / "batch.json", payload)
    return payload


def _run_generation_batch_passk(
    request: GenerationBatchRequest,
    *,
    log_fn: LogFn | None = None,
    on_generation_done: GenerationDoneFn | None = None,
) -> dict:
    """Run pass@k LLM generation and write op-level/candidate-level summaries."""
    records: list[dict | None] = [None] * len(request.op_names)
    candidates_by_op: dict[str, list[dict]] = {op_name: [] for op_name in request.op_names}
    max_workers = max(1, request.llm_workers)
    total_candidates = len(request.op_names) * request.samples_per_op
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        flattened_index = 0
        for op_index, op_name in enumerate(request.op_names):
            op_run_id = f"{request.run_id}_{op_name}"
            op_run_dir = category_op_run_dir(request.run_dir, op_name)
            for candidate_id in range(request.samples_per_op):
                device_id = request.device_ids[flattened_index % len(request.device_ids)]
                candidate_run_id = f"{op_run_id}_candidate_{candidate_id:02d}"
                candidate_run_dir = _candidate_dir(op_run_dir, candidate_id)
                future = executor.submit(
                    _generate_one,
                    request=request,
                    op_name=op_name,
                    device_id=device_id,
                    op_run_id=candidate_run_id,
                    op_run_dir=candidate_run_dir,
                    log_fn=log_fn,
                    candidate_id=candidate_id,
                    samples_per_op=request.samples_per_op,
                )
                futures[future] = (
                    op_index,
                    op_name,
                    device_id,
                    op_run_id,
                    op_run_dir,
                    candidate_id,
                )
                flattened_index += 1

        for future in as_completed(futures):
            op_index, op_name, device_id, op_run_id, op_run_dir, candidate_id = futures[future]
            category = get_operator_spec(op_name)["category"]
            try:
                generation = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                generation = _generation_failure(
                    request=request,
                    op_run_id=f"{op_run_id}_candidate_{candidate_id:02d}",
                    op_name=op_name,
                    op_run_dir=_candidate_dir(op_run_dir, candidate_id),
                    error=exc,
                    candidate_id=candidate_id,
                    samples_per_op=request.samples_per_op,
                )
            candidate_record = _candidate_generation_record(
                op_name=op_name,
                category=category,
                device_id=device_id,
                op_run_dir=op_run_dir,
                candidate_id=candidate_id,
                generation=generation,
            )
            candidates_by_op[op_name].append(candidate_record)
            if on_generation_done is not None:
                on_generation_done(candidate_record, generation)

            if len(candidates_by_op[op_name]) == request.samples_per_op:
                ordered_candidates = sorted(
                    candidates_by_op[op_name],
                    key=lambda item: int(item.get("candidate_id", 0)),
                )
                summary = _op_passk_generation_summary(
                    request=request,
                    op_name=op_name,
                    category=category,
                    op_run_id=op_run_id,
                    op_run_dir=op_run_dir,
                    candidates=ordered_candidates,
                )
                records[op_index] = _op_passk_record(
                    op_name=op_name,
                    category=category,
                    device_id=request.device_ids[op_index % len(request.device_ids)],
                    op_run_dir=op_run_dir,
                    summary=summary,
                )

    finalized = [
        record
        if record is not None
        else {
            "op_name": request.op_names[index],
            "category": get_operator_spec(request.op_names[index])["category"],
            "device_id": request.device_ids[index % len(request.device_ids)],
            "samples_per_op": request.samples_per_op,
            "generation_success": False,
            "success": False,
            "error_message": "Operator did not produce pass@k generation records.",
        }
        for index, record in enumerate(records)
    ]
    payload = _batch_payload(request=request, records=finalized)
    payload["total_candidates"] = total_candidates
    payload["successful_candidates"] = sum(
        int(record.get("successful_candidates") or 0)
        for record in finalized
    )
    payload["failed_candidates"] = total_candidates - payload["successful_candidates"]
    payload["temperature"] = PASSK_TEMPERATURE
    write_json(request.run_dir / "batch.json", payload)
    return payload
