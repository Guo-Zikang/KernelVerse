"""CLI entrypoint for generated CUDA evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.batch_runner import run_evaluation_tasks, run_single_evaluation_task
from evaluator.compilation_runner import (
    compilation_success,
    run_compilation_tasks,
    run_single_compilation_task,
)
from evaluator.tasks import (
    common_task_value,
    default_compile_run_id,
    default_reeval_run_id,
    discover_source_tasks,
    normal_task,
    repo_root,
    select_source_tasks,
    single_normal_task,
    source_run_dir,
)
from evaluator.workflow import evaluation_success
from operators.registry import resolve_operator_selection
from utils.progress_logger import ProgressLogger


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile or evaluate generated operator code.")
    selector_group = parser.add_mutually_exclusive_group(required=False)
    selector_group.add_argument(
        "--op",
        help=(
            "Comma-separated operator names to compile or evaluate, for example: "
            "addmm,bmm,conv2d."
        ),
    )
    selector_group.add_argument(
        "--category",
        help="Operator category to compile or evaluate, for example: attention, gemm, or tensor_ops.",
    )
    selector_group.add_argument(
        "--all",
        action="store_true",
        help="Compile or evaluate every registered operator.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing generation run id under runs/ to compile or re-evaluate.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile generated code and check for ModelNew without running correctness or performance.",
    )
    parser.add_argument(
        "--shape_mode",
        default=None,
        choices=["single", "family", "random"],
        help="Shape execution mode. Required unless --run-id is used.",
    )
    parser.add_argument(
        "--shape_name",
        default=None,
        help="Single-shape template name. Used when --shape_mode single.",
    )
    parser.add_argument(
        "--shape_family",
        default=None,
        help="Shape family name for family mode, or all to evaluate every family.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Target backend language. Defaults to cuda unless --run-id infers it from metadata.",
    )
    parser.add_argument(
        "--generated-path",
        default=None,
        help="Generated .txt code path. Required with single --op.",
    )
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="Directory containing <op>.txt generated code files for --category or --all.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory where compilation.json or evaluation.json files will be written.",
    )
    parser.add_argument(
        "--device-ids",
        type=_parse_device_ids,
        default=_parse_device_ids("0"),
        help="Comma-separated accelerator device ids used for compilation or evaluation.",
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
        "--resume",
        action="store_true",
        help="Resume from existing per-operator result files in --run-dir.",
    )
    return parser


def _log_run_start(logger: ProgressLogger, *, run_id: str, run_dir: Path, selector: str) -> float:
    start = time.monotonic()
    logger.log(
        "run",
        "start",
        message=f"run_id={run_id} run_dir={run_dir} selector={selector} total_ops={logger.total_ops}",
    )
    return start


def _log_run_done(logger: ProgressLogger, *, run_id: str, start: float) -> None:
    logger.log(
        "run",
        "done",
        message=f"run_id={run_id} elapsed={time.monotonic() - start:.1f}s",
    )


def _run_batch_evaluation(
    args: argparse.Namespace,
    run_dir: Path,
    op_names: list[str],
    logger: ProgressLogger,
) -> dict:
    tasks = [normal_task(args, run_dir, op_name) for op_name in op_names]
    if args.all:
        selector = {"type": "all", "value": "all"}
    elif args.category:
        selector = {"type": "category", "value": args.category}
    else:
        selector = {"type": "ops", "value": op_names}
    summary_context = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "selector": selector,
        "shape_mode": args.shape_mode,
        "shape_name": args.shape_name,
        "shape_family": args.shape_family,
        "language": args.language,
    }
    return run_evaluation_tasks(
        args=args,
        run_dir=run_dir,
        tasks=tasks,
        summary_context=summary_context,
        log_fn=logger,
    )


def _run_batch_compilation(
    args: argparse.Namespace,
    run_dir: Path,
    op_names: list[str],
    logger: ProgressLogger,
) -> dict:
    tasks = [normal_task(args, run_dir, op_name) for op_name in op_names]
    if args.all:
        selector = {"type": "all", "value": "all"}
    elif args.category:
        selector = {"type": "category", "value": args.category}
    else:
        selector = {"type": "ops", "value": op_names}
    summary_context = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "selector": selector,
        "mode": "compile_only",
        "language": args.language,
    }
    return run_compilation_tasks(
        args=args,
        run_dir=run_dir,
        tasks=tasks,
        summary_context=summary_context,
        log_fn=logger,
    )


def _run_id_evaluation(
    args: argparse.Namespace,
    source_run_dir: Path,
    run_dir: Path,
    selected_tasks: list,
    selector: dict,
    logger: ProgressLogger,
) -> dict:
    summary_context = {
        "source_run_id": args.run_id,
        "source_run_dir": str(source_run_dir),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "selector": selector,
        "shape_mode": common_task_value(selected_tasks, "shape_mode"),
        "language": common_task_value(selected_tasks, "language"),
    }
    return run_evaluation_tasks(
        args=args,
        run_dir=run_dir,
        tasks=selected_tasks,
        summary_context=summary_context,
        log_fn=logger,
    )


def _run_id_compilation(
    args: argparse.Namespace,
    source_run_dir: Path,
    run_dir: Path,
    selected_tasks: list,
    selector: dict,
    logger: ProgressLogger,
) -> dict:
    summary_context = {
        "source_run_id": args.run_id,
        "source_run_dir": str(source_run_dir),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "selector": selector,
        "mode": "compile_only",
        "language": common_task_value(selected_tasks, "language"),
    }
    return run_compilation_tasks(
        args=args,
        run_dir=run_dir,
        tasks=selected_tasks,
        summary_context=summary_context,
        log_fn=logger,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir pointing to an existing result directory.")

    if args.run_id:
        forbidden = []
        for option_name, value in (
            ("--shape_mode", args.shape_mode),
            ("--shape_name", args.shape_name),
            ("--shape_family", args.shape_family),
            ("--language", args.language),
            ("--generated-path", args.generated_path),
            ("--generated-dir", args.generated_dir),
            ("--random-count", args.random_count),
            ("--random-max-attempts", args.random_max_attempts),
        ):
            if value is not None:
                forbidden.append(option_name)
        if forbidden:
            parser.error(
                "--run-id infers generated paths and shape metadata; do not pass "
                + ", ".join(forbidden)
                + "."
            )
        source_dir = source_run_dir(args.run_id)
        if not source_dir.exists():
            parser.error(f"Source run id not found: {args.run_id} ({source_dir})")
        run_dir = (
            Path(args.run_dir).resolve()
            if args.run_dir
            else repo_root() / "runs" / (
                default_compile_run_id(args.run_id)
                if args.compile_only
                else default_reeval_run_id(args.run_id)
            )
        )
        if run_dir.resolve() == source_dir.resolve():
            parser.error("--run-dir must not point to the source run directory in --run-id mode.")
        try:
            source_tasks, source_selector = discover_source_tasks(source_dir)
            selected_tasks, selector = select_source_tasks(
                args=args,
                source_tasks=source_tasks,
                source_selector=source_selector,
            )
        except ValueError as exc:
            parser.error(str(exc))
        logger = ProgressLogger(total_ops=len(selected_tasks), log_path=run_dir / "run.log")
        run_start = _log_run_start(
            logger,
            run_id=run_dir.name,
            run_dir=run_dir,
            selector=f"run_id={args.run_id}",
        )
        try:
            if args.compile_only:
                result = _run_id_compilation(
                    args,
                    source_dir,
                    run_dir,
                    selected_tasks,
                    selector,
                    logger,
                )
            else:
                result = _run_id_evaluation(
                    args,
                    source_dir,
                    run_dir,
                    selected_tasks,
                    selector,
                    logger,
                )
        finally:
            _log_run_done(logger, run_id=run_dir.name, start=run_start)
        return 0 if result.get("success") else 1

    if not (args.op or args.category or args.all):
        parser.error("Specify exactly one of --op, --category, --all, or use --run-id.")
    if not args.shape_mode and not args.compile_only:
        parser.error("--shape_mode is required unless --run-id is used.")
    if not args.run_dir:
        parser.error("--run-dir is required unless --run-id is used.")

    args.shape_mode = args.shape_mode or "single"
    args.shape_name = args.shape_name or "default"
    args.shape_family = args.shape_family or "all"
    args.language = args.language or "cuda"

    run_dir = Path(args.run_dir).resolve()
    try:
        op_names = resolve_operator_selection(
            op_name=args.op,
            category=args.category,
            all_ops=args.all,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.generated_path and (not args.op or len(op_names) != 1):
        parser.error("--generated-path is only supported with one operator in --op.")

    if args.op and len(op_names) == 1:
        if not args.generated_path:
            parser.error("--generated-path is required when --op selects one operator.")
        if args.generated_dir:
            parser.error(
                "--generated-dir is only supported with multiple operators, "
                "--category, or --all."
            )
        op_name = op_names[0]
        task = single_normal_task(args, Path(args.generated_path).resolve(), op_name)
        logger = ProgressLogger(total_ops=1, log_path=run_dir / "run.log")
        run_start = _log_run_start(
            logger,
            run_id=run_dir.name,
            run_dir=run_dir,
            selector=f"op={op_name}",
        )
        result = None
        try:
            if args.compile_only:
                result = run_single_compilation_task(
                    args=args,
                    run_dir=run_dir,
                    task=task,
                    log_fn=logger,
                )
            else:
                result = run_single_evaluation_task(
                    args=args,
                    run_dir=run_dir,
                    task=task,
                    log_fn=logger,
                )
        finally:
            _log_run_done(logger, run_id=run_dir.name, start=run_start)
        return 0 if (
            compilation_success(result)
            if args.compile_only
            else evaluation_success(result)
        ) else 1

    if args.op and not args.generated_dir:
        parser.error("--generated-dir is required when --op selects multiple operators.")

    logger = ProgressLogger(total_ops=len(op_names), log_path=run_dir / "run.log")
    if args.all:
        selector_label = "all"
    elif args.category:
        selector_label = f"category={args.category}"
    else:
        selector_label = "ops=" + ",".join(op_names)
    run_start = _log_run_start(
        logger,
        run_id=run_dir.name,
        run_dir=run_dir,
        selector=selector_label,
    )
    try:
        result = (
            _run_batch_compilation(args, run_dir, op_names, logger)
            if args.compile_only
            else _run_batch_evaluation(args, run_dir, op_names, logger)
        )
    finally:
        _log_run_done(logger, run_id=run_dir.name, start=run_start)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
