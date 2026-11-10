"""CLI entrypoint for CUDA code generation."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codegen.batch_runner import GenerationBatchRequest, run_generation_batch
from codegen.generators import GenerationRequest, run_generation_with_retries
from codegen.retry_failed import (
    FailedGenerationRetryRequest,
    discover_failed_generation_tasks,
    retry_failed_generations,
)
from operators.registry import resolve_operator_selection
from utils.progress_logger import ProgressLogger


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_run_id(label: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1.")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0.")
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
    parser = argparse.ArgumentParser(description="Generate accelerator code for operators.")
    selector_group = parser.add_mutually_exclusive_group(required=True)
    selector_group.add_argument(
        "--op",
        help=(
            "Comma-separated operator names to generate, for example: "
            "addmm,bmm,conv2d."
        ),
    )
    selector_group.add_argument(
        "--category",
        help="Operator category to generate, for example: attention, gemm, or tensor_ops.",
    )
    selector_group.add_argument(
        "--all",
        action="store_true",
        help="Generate code for every registered operator.",
    )
    selector_group.add_argument(
        "--retry-failed-run-dir",
        help=(
            "Retry each generation.json with success=false in the given existing "
            "run directory."
        ),
    )
    parser.add_argument(
        "--shape_mode",
        choices=["single", "family", "random"],
        help=(
            "Shape context mode used in the generation prompt. Required unless "
            "--retry-failed-run-dir is used."
        ),
    )
    parser.add_argument(
        "--shape_name",
        default="default",
        help="Single-shape template name. Used when --shape_mode single.",
    )
    parser.add_argument(
        "--shape_family",
        default="all",
        help="Shape family name for family mode, or all to include every family.",
    )
    parser.add_argument(
        "--language",
        default="cuda",
        help="Target backend language for the generated implementation.",
    )
    parser.add_argument(
        "--device-ids",
        type=_parse_device_ids,
        default=None,
        help=(
            "Comma-separated accelerator device ids used when collecting runtime "
            "environment information. Retry mode reuses recorded devices unless "
            "this option is provided."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory for generation.json, generated code, and artifacts.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the generated code file already exists.",
    )
    parser.add_argument(
        "--env-path",
        default=None,
        help="Path to the .env file containing LLM API configuration.",
    )
    parser.add_argument(
        "--random-count",
        type=_positive_int,
        default=None,
        help="Number of random shapes to include when --shape_mode random.",
    )
    parser.add_argument(
        "--random-max-attempts",
        type=_positive_int,
        default=None,
        help="Maximum attempts for generating unique random shapes.",
    )
    parser.add_argument(
        "--llm-workers",
        type=_positive_int,
        default=200,
        help="Maximum concurrent LLM generation requests in batch mode.",
    )
    parser.add_argument(
        "--generation-retries",
        type=_non_negative_int,
        default=5,
        help=(
            "Maximum retries after an initial generation failure. Use 0 to "
            "disable retries."
        ),
    )
    parser.add_argument(
        "--samples-per-op",
        type=_positive_int,
        default=1,
        help="Number of independent generated candidates per operator. Use 10 for pass@10.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.retry_failed_run_dir:
        if args.run_dir is not None:
            parser.error("--run-dir cannot be combined with --retry-failed-run-dir.")
        retry_request = FailedGenerationRetryRequest(
            run_dir=Path(args.retry_failed_run_dir),
            llm_workers=args.llm_workers,
            generation_retries=args.generation_retries,
            env_path=Path(args.env_path) if args.env_path else None,
            device_ids=args.device_ids,
        )
        try:
            tasks = discover_failed_generation_tasks(retry_request)
        except ValueError as exc:
            parser.error(str(exc))
        logger = ProgressLogger(
            total_ops=max(1, len(tasks)),
            log_path=retry_request.run_dir.resolve() / "run.log",
        )
        if not tasks:
            logger.log(
                "run",
                "done",
                message="No failed generation.json files found.",
            )
            return 0
        result = retry_failed_generations(
            retry_request,
            tasks=tasks,
            log_fn=logger,
        )
        return 0 if result.get("success") else 1

    if args.shape_mode is None:
        parser.error("--shape_mode is required for code generation.")
    device_ids = args.device_ids or [0]
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
    if args.op and len(op_names) == 1 and args.samples_per_op == 1:
        op_name = op_names[0]
        start = time.monotonic()
        logger.log("codegen", "start", op_name=op_name, device_id=device_ids[0])
        generation_request = GenerationRequest(
            run_id=run_id,
            op_name=op_name,
            shape_mode=args.shape_mode,
            shape_name=args.shape_name,
            shape_family=args.shape_family,
            language=args.language,
            run_dir=run_dir,
            device_id=device_ids[0],
            env_path=Path(args.env_path) if args.env_path else None,
            no_overwrite=args.no_overwrite,
            random_count=args.random_count,
            random_max_attempts=args.random_max_attempts,
        )
        result = run_generation_with_retries(
            generation_request,
            generation_retries=args.generation_retries,
            on_retry=lambda retry, total, _result: logger.log(
                "codegen",
                "retry",
                op_name=op_name,
                device_id=device_ids[0],
                success=False,
                message=f"retry={retry}/{total}",
            ),
        )
        logger.log(
            "codegen",
            "done",
            op_name=op_name,
            device_id=device_ids[0],
            success=bool(result.get("success")),
            elapsed=time.monotonic() - start,
        )
        return 0 if result.get("success") else 1

    result = run_generation_batch(
        GenerationBatchRequest(
            run_id=run_id,
            run_dir=run_dir,
            op_names=op_names,
            selector=_selector_payload(args, op_names),
            shape_mode=args.shape_mode,
            shape_name=args.shape_name,
            shape_family=args.shape_family,
            language=args.language,
            device_ids=device_ids,
            llm_workers=args.llm_workers,
            env_path=Path(args.env_path) if args.env_path else None,
            no_overwrite=args.no_overwrite,
            random_count=args.random_count,
            random_max_attempts=args.random_max_attempts,
            samples_per_op=args.samples_per_op,
            generation_retries=args.generation_retries,
        ),
        log_fn=logger,
    )
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
