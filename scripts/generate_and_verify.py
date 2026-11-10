"""Create a run directory, generate CUDA code, then evaluate it."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codegen.artifacts import write_json
from codegen.batch_runner import GenerationBatchRequest, run_generation_batch
from codegen.generators import GenerationRequest, run_generation
from evaluator.batch_runner import StreamingEvaluationScheduler, run_single_evaluation_task
from evaluator.compilation_runner import (
    StreamingCompilationScheduler,
    compilation_success,
    run_single_compilation_task,
)
from evaluator.tasks import (
    discover_source_tasks,
    task_from_generation,
)
from evaluator.workflow import evaluation_success
from operators.registry import resolve_operator_selection
from utils.progress_logger import ProgressLogger
from utils.run_paths import op_run_dir as category_op_run_dir


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1.")
    return parsed


def _parse_device_ids(value: str) -> list[int]:
    if not value:
        raise argparse.ArgumentTypeError("--device-ids cannot be empty.")
    try:
        device_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--device-ids must be comma-separated integers.") from exc
    if not device_ids:
        raise argparse.ArgumentTypeError("--device-ids cannot be empty.")
    if any(device_id < 0 for device_id in device_ids):
        raise argparse.ArgumentTypeError("--device-ids must be non-negative.")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("--device-ids cannot contain duplicates.")
    return device_ids


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_run_id(label: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"


def _selector_label(args: argparse.Namespace, op_names: list[str]) -> str:
    if args.all:
        return "all"
    if args.category:
        return f"category_{args.category}"
    if len(op_names) == 1:
        return op_names[0]
    return f"ops_{len(op_names)}"


def _selector_payload(args: argparse.Namespace, op_names: list[str]) -> dict:
    if args.all:
        return {"type": "all", "value": "all"}
    if args.category:
        return {"type": "category", "value": args.category}
    if len(op_names) == 1:
        return {"type": "op", "value": op_names[0]}
    return {"type": "ops", "value": op_names}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate operator code, then compile or evaluate it in one run directory."
    )
    selector_group = parser.add_mutually_exclusive_group(required=True)
    selector_group.add_argument(
        "--op",
        help=(
            "Comma-separated operator names to generate, then compile or evaluate, for example: "
            "addmm,bmm,conv2d."
        ),
    )
    selector_group.add_argument(
        "--category",
        help="Operator category to generate, then compile or evaluate.",
    )
    selector_group.add_argument(
        "--all",
        action="store_true",
        help="Generate every registered operator, then compile or evaluate it.",
    )
    parser.add_argument(
        "--shape_mode",
        required=True,
        choices=["single", "family", "random"],
        help="Shape execution mode: single, family, or random.",
    )
    parser.add_argument(
        "--shape_name",
        default="default",
        help="Single-shape template name. Used when --shape_mode single.",
    )
    parser.add_argument(
        "--shape_family",
        default="all",
        help="Shape family name for family mode, or all to evaluate every family.",
    )
    parser.add_argument(
        "--language",
        default="cuda",
        help="Target backend language for generated code.",
    )
    parser.add_argument(
        "--verify-trials",
        type=_positive_int,
        default=5,
        help="Number of randomized correctness trials per shape.",
    )
    parser.add_argument(
        "--perf-trials",
        type=_positive_int,
        default=5,
        help="Number of timing trials per shape case.",
    )
    parser.add_argument(
        "--random-count",
        type=_positive_int,
        default=None,
        help="Number of random shape combinations to generate for --shape_mode random.",
    )
    parser.add_argument(
        "--random-max-attempts",
        type=_positive_int,
        default=None,
        help="Maximum attempts for generating unique random shapes.",
    )
    parser.add_argument(
        "--device-ids",
        type=_parse_device_ids,
        default=_parse_device_ids("0"),
        help="Comma-separated accelerator device ids used for generation context and execution.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory. Defaults to runs/<timestamp>_<selector>.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if a generated code file already exists.",
    )
    parser.add_argument(
        "--env-path",
        default=None,
        help="Path to the .env file containing LLM API configuration.",
    )
    parser.add_argument(
        "--llm-workers",
        type=_positive_int,
        default=200,
        help="Maximum concurrent LLM generation requests in batch mode.",
    )
    parser.add_argument(
        "--samples-per-op",
        type=_positive_int,
        default=1,
        help="Number of independent generated candidates per operator. Use 10 for pass@10.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Generate and compile code without running correctness or performance evaluation.",
    )
    return parser


def _run_single(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    logger: ProgressLogger,
    op_name: str,
) -> dict:
    codegen_start = time.monotonic()
    logger.log("codegen", "start", op_name=op_name, device_id=args.device_ids[0])
    generation = run_generation(
        GenerationRequest(
            run_id=run_id,
            op_name=op_name,
            shape_mode=args.shape_mode,
            shape_name=args.shape_name,
            shape_family=args.shape_family,
            language=args.language,
            run_dir=run_dir,
            device_id=args.device_ids[0],
            env_path=Path(args.env_path) if args.env_path else None,
            no_overwrite=args.no_overwrite,
            random_count=args.random_count,
            random_max_attempts=args.random_max_attempts,
        )
    )
    logger.log(
        "codegen",
        "done",
        op_name=op_name,
        device_id=args.device_ids[0],
        success=bool(generation.get("success")),
        elapsed=time.monotonic() - codegen_start,
    )

    source_tasks, _ = discover_source_tasks(run_dir)
    task = source_tasks[0]
    if args.compile_only:
        result = run_single_compilation_task(
            args=args,
            run_dir=run_dir,
            task=task,
            log_fn=logger,
        )
        result_key = "compilation"
        result_json_key = "compilation_json"
        result_json = run_dir / "compilation.json"
        result_success = compilation_success(result)
    else:
        result = run_single_evaluation_task(
            args=args,
            run_dir=run_dir,
            task=task,
            log_fn=logger,
        )
        result_key = "evaluation"
        result_json_key = "evaluation_json"
        result_json = run_dir / "evaluation.json"
        result_success = evaluation_success(result)

    success = bool(generation.get("success")) and result_success
    payload = {
        "success": success,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "generation_json": str(run_dir / "generation.json"),
        result_json_key: str(result_json),
        "generation": generation,
        result_key: result,
    }
    if not success:
        payload["error_message"] = (
            generation.get("error_message")
            if not generation.get("success")
            else result.get("error_message") or (
                "Compilation failed." if args.compile_only else "Evaluation failed."
            )
        )
    return payload


def _evaluation_summary_context_from_args(
    *,
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
    op_names: list[str],
) -> dict:
    context = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "selector": _selector_payload(args, op_names),
        "shape_mode": args.shape_mode,
        "language": args.language,
    }
    if args.shape_mode == "single":
        context["shape_name"] = args.shape_name
    elif args.shape_mode == "family":
        context["shape_family"] = args.shape_family
    return context


def _compilation_summary_context_from_args(
    *,
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
    op_names: list[str],
) -> dict:
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "selector": _selector_payload(args, op_names),
        "mode": "compile_only",
        "language": args.language,
    }


def _batch_metadata(args: argparse.Namespace) -> dict:
    return {
        "shape_mode": args.shape_mode,
        "shape_name": args.shape_name,
        "shape_family": args.shape_family,
        "language": args.language,
    }


def _merge_batch_payloads(
    *,
    generation_payload: dict,
    evaluation_payload: dict,
    run_dir: Path,
) -> dict:
    evaluation_success_by_op = {
        record.get("op_name"): bool(record.get("success"))
        for record in evaluation_payload.get("ops", [])
        if record.get("op_name")
    }
    evaluation_by_op = {
        record["op_name"]: record
        for record in evaluation_payload.get("ops", [])
        if record.get("op_name")
    }
    combined_records = []
    for generation_record in generation_payload.get("ops", []):
        op_name = generation_record.get("op_name")
        evaluation_record = evaluation_by_op.get(op_name) or {}
        generation_success = bool(generation_record.get("generation_success"))
        evaluation_success_value = evaluation_success_by_op.get(op_name, False)
        success = generation_success and evaluation_success_value
        error_type = None
        error_message = None
        if not generation_success:
            error_type = generation_record.get("error_type")
            error_message = generation_record.get("error_message")
        elif not evaluation_success_value:
            error_type = evaluation_record.get("error_type")
            error_message = evaluation_record.get("error_message") or "Evaluation failed."
        reward_hacking_risk_level = (
            evaluation_record.get("reward_hacking_risk_level")
            or generation_record.get("reward_hacking_risk_level")
        )
        op_run_dir = Path(
            generation_record.get("run_dir")
            or category_op_run_dir(run_dir, str(op_name))
        )
        evaluation_candidates_by_id = {
            int(candidate["candidate_id"]): candidate
            for candidate in (evaluation_record.get("candidates") or [])
            if isinstance(candidate, dict) and candidate.get("candidate_id") is not None
        }
        merged_candidates = []
        for generation_candidate in generation_record.get("candidates") or []:
            if not isinstance(generation_candidate, dict):
                continue
            candidate_id = generation_candidate.get("candidate_id")
            evaluation_candidate = {}
            if candidate_id is not None:
                try:
                    evaluation_candidate = evaluation_candidates_by_id.get(int(candidate_id), {})
                except (TypeError, ValueError):
                    evaluation_candidate = {}
            candidate_success = bool(generation_candidate.get("generation_success")) and bool(
                evaluation_candidate.get("success")
            )
            merged_candidates.append(
                {
                    **generation_candidate,
                    "evaluation_json": evaluation_candidate.get("evaluation_json"),
                    "evaluation_success": bool(evaluation_candidate.get("success")),
                    "success": candidate_success,
                    "error_type": (
                        generation_candidate.get("error_type")
                        if not generation_candidate.get("generation_success")
                        else evaluation_candidate.get("error_type")
                    ),
                    "error_message": (
                        generation_candidate.get("error_message")
                        if not generation_candidate.get("generation_success")
                        else evaluation_candidate.get("error_message")
                    ),
                }
            )
        combined_record = {
            **generation_record,
            "evaluation_json": evaluation_record.get(
                "evaluation_json",
                str(op_run_dir / "evaluation.json"),
            ),
            "evaluation_success": evaluation_success_value,
            "reward_hacking_risk_level": reward_hacking_risk_level,
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
        }
        if merged_candidates:
            combined_record["candidates"] = merged_candidates
        for key in (
            "pass_at_10_raw",
            "passed_candidates",
            "failed_candidates",
            "total_candidates",
            "samples_per_op",
        ):
            if key in evaluation_record:
                combined_record[key] = evaluation_record[key]
        combined_records.append(combined_record)

    successful_ops = sum(1 for record in combined_records if record.get("success"))
    generation_successful_ops = sum(
        1 for record in combined_records if record.get("generation_success")
    )
    evaluation_successful_ops = sum(
        1 for record in combined_records if record.get("evaluation_success")
    )
    payload = {
        **generation_payload,
        "success": successful_ops == len(combined_records),
        "generation_success": bool(generation_payload.get("success")),
        "evaluation_success": bool(evaluation_payload.get("success")),
        "evaluation_json": str(run_dir / "evaluation.json"),
        "successful_ops": successful_ops,
        "failed_ops": len(combined_records) - successful_ops,
        "generation_successful_ops": generation_successful_ops,
        "evaluation_successful_ops": evaluation_successful_ops,
        "ops": combined_records,
    }
    for key in (
        "samples_per_op",
        "pass_at_10_raw_ops",
        "pass_at_10_raw_rate",
        "total_candidates",
        "passed_candidates",
        "failed_candidates",
    ):
        if key in evaluation_payload:
            payload[key] = evaluation_payload[key]
    write_json(run_dir / "batch.json", payload)
    return payload


def _merge_batch_compilation_payloads(
    *,
    generation_payload: dict,
    compilation_payload: dict,
    run_dir: Path,
) -> dict:
    compilation_by_op = {
        record["op_name"]: record
        for record in compilation_payload.get("ops", [])
        if record.get("op_name")
    }
    combined_records = []
    for generation_record in generation_payload.get("ops", []):
        op_name = generation_record.get("op_name")
        compilation_record = compilation_by_op.get(op_name) or {}
        generation_success = bool(generation_record.get("generation_success"))
        compilation_success_value = bool(compilation_record.get("success"))
        op_run_dir = Path(
            generation_record.get("run_dir")
            or category_op_run_dir(run_dir, str(op_name))
        )

        compilation_candidates_by_id = {
            int(candidate["candidate_id"]): candidate
            for candidate in (compilation_record.get("candidates") or [])
            if isinstance(candidate, dict) and candidate.get("candidate_id") is not None
        }
        merged_candidates = []
        for generation_candidate in generation_record.get("candidates") or []:
            if not isinstance(generation_candidate, dict):
                continue
            candidate_id = generation_candidate.get("candidate_id")
            compiled_candidate = {}
            if candidate_id is not None:
                try:
                    compiled_candidate = compilation_candidates_by_id.get(int(candidate_id), {})
                except (TypeError, ValueError):
                    compiled_candidate = {}
            candidate_success = bool(generation_candidate.get("generation_success")) and bool(
                compiled_candidate.get("success")
            )
            merged_candidates.append(
                {
                    **generation_candidate,
                    "compilation_json": compiled_candidate.get("compilation_json"),
                    "compilation_success": bool(compiled_candidate.get("success")),
                    "compiled": bool(compiled_candidate.get("compiled")),
                    "success": candidate_success,
                    "error_type": (
                        generation_candidate.get("error_type")
                        if not generation_candidate.get("generation_success")
                        else compiled_candidate.get("error_type")
                    ),
                    "error_message": (
                        generation_candidate.get("error_message")
                        if not generation_candidate.get("generation_success")
                        else compiled_candidate.get("error_message")
                    ),
                }
            )

        success = generation_success and compilation_success_value
        combined_record = {
            **generation_record,
            "compilation_json": compilation_record.get(
                "compilation_json",
                str(op_run_dir / "compilation.json"),
            ),
            "compilation_success": compilation_success_value,
            "success": success,
            "error_type": (
                generation_record.get("error_type")
                if not generation_success
                else compilation_record.get("error_type")
            ),
            "error_message": (
                generation_record.get("error_message")
                if not generation_success
                else compilation_record.get("error_message")
            ),
        }
        if merged_candidates:
            combined_record["candidates"] = merged_candidates
        for key in (
            "samples_per_op",
            "total_candidates",
            "compiled_candidates",
            "failed_candidates",
        ):
            if key in compilation_record:
                combined_record[key] = compilation_record[key]
        combined_records.append(combined_record)

    successful_ops = sum(1 for record in combined_records if record.get("success"))
    payload = {
        **generation_payload,
        "success": successful_ops == len(combined_records),
        "generation_success": bool(generation_payload.get("success")),
        "compilation_success": bool(compilation_payload.get("success")),
        "compilation_json": str(run_dir / "compilation.json"),
        "successful_ops": successful_ops,
        "failed_ops": len(combined_records) - successful_ops,
        "generation_successful_ops": sum(
            1 for record in combined_records if record.get("generation_success")
        ),
        "compilation_successful_ops": sum(
            1 for record in combined_records if record.get("compilation_success")
        ),
        "ops": combined_records,
    }
    for key in (
        "samples_per_op",
        "total_candidates",
        "compiled_candidates",
        "failed_candidates",
    ):
        if key in compilation_payload:
            payload[key] = compilation_payload[key]
    write_json(run_dir / "batch.json", payload)
    return payload


def _run_batch(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    op_names: list[str],
    logger: ProgressLogger,
) -> dict:
    scheduler_class = (
        StreamingCompilationScheduler
        if args.compile_only
        else StreamingEvaluationScheduler
    )
    summary_context = (
        _compilation_summary_context_from_args(
            run_id=run_id,
            run_dir=run_dir,
            args=args,
            op_names=op_names,
        )
        if args.compile_only
        else _evaluation_summary_context_from_args(
            run_id=run_id,
            run_dir=run_dir,
            args=args,
            op_names=op_names,
        )
    )
    scheduler = scheduler_class(
        args=args,
        run_dir=run_dir,
        summary_context=summary_context,
        log_fn=logger,
    )
    batch_metadata = _batch_metadata(args)

    def on_generation_done(record: dict, generation: dict) -> None:
        op_name = str(record["op_name"])
        device_id = int(record["device_id"])
        op_run_dir = Path(record["run_dir"])
        generated_path = Path(record["generated_path"])
        task = task_from_generation(
            op_name=op_name,
            generated_path=generated_path,
            generation=generation,
            batch_metadata=batch_metadata,
        )
        if not record.get("generation_success"):
            error_message = generation.get("error_message") or "Code generation failed."
            scheduler.record_failure(
                task,
                device_id=device_id,
                op_run_dir=op_run_dir,
                error=RuntimeError(error_message),
                error_type=generation.get("error_type") or "GenerationFailed",
            )
            return
        scheduler.submit(task, device_id=device_id, op_run_dir=op_run_dir)

    generation_payload = run_generation_batch(
        GenerationBatchRequest(
            run_id=run_id,
            run_dir=run_dir,
            op_names=op_names,
            selector=_selector_payload(args, op_names),
            shape_mode=args.shape_mode,
            shape_name=args.shape_name,
            shape_family=args.shape_family,
            language=args.language,
            device_ids=args.device_ids,
            llm_workers=args.llm_workers,
            env_path=Path(args.env_path) if args.env_path else None,
            no_overwrite=args.no_overwrite,
            random_count=args.random_count,
            random_max_attempts=args.random_max_attempts,
            samples_per_op=args.samples_per_op,
        ),
        log_fn=logger,
        on_generation_done=on_generation_done,
    )
    result_payload = scheduler.finish()
    if args.compile_only:
        return _merge_batch_compilation_payloads(
            generation_payload=generation_payload,
            compilation_payload=result_payload,
            run_dir=run_dir,
        )
    return _merge_batch_payloads(
        generation_payload=generation_payload,
        evaluation_payload=result_payload,
        run_dir=run_dir,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        op_names = resolve_operator_selection(
            op_name=args.op,
            category=args.category,
            all_ops=args.all,
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = _repo_root()
    run_id = (
        Path(args.run_dir).name
        if args.run_dir
        else _default_run_id(_selector_label(args, op_names))
    )
    run_dir = Path(args.run_dir).resolve() if args.run_dir else root / "runs" / run_id
    logger = ProgressLogger(
        total_ops=len(op_names) * args.samples_per_op,
        log_path=run_dir / "run.log",
    )

    run_start = time.monotonic()
    logger.log(
        "run",
        "start",
        message=(
            f"run_id={run_id} run_dir={run_dir} "
            f"selector={_selector_label(args, op_names)} total_ops={len(op_names)}"
        ),
    )
    try:
        if args.op and len(op_names) == 1 and args.samples_per_op == 1:
            payload = _run_single(args, run_id, run_dir, logger, op_names[0])
        else:
            payload = _run_batch(args, run_id, run_dir, op_names, logger)
    finally:
        logger.log(
            "run",
            "done",
            message=f"run_id={run_id} elapsed={time.monotonic() - run_start:.1f}s",
        )

    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
