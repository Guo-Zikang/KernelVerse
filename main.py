"""CLI entrypoint for the operator benchmark framework."""

import argparse
import json

from operators.registry import resolve_operator_selection
from runner.benchmark_runner import run_benchmark


def parse_workers(value: str) -> int:
    """Parse and validate the worker-process count."""
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("--workers must be >= 1.")
    return workers


def parse_positive_int(value: str) -> int:
    """Parse and validate a positive integer CLI argument."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1.")
    return parsed_value


def parse_device_ids(value: str) -> list[int]:
    """Parse and validate a comma-separated device-id list."""
    if not value:
        raise argparse.ArgumentTypeError("--device-ids cannot be empty.")

    try:
        device_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--device-ids must be a comma-separated integer list."
        ) from error

    if not device_ids:
        raise argparse.ArgumentTypeError("--device-ids cannot be empty.")
    invalid_ids = [device_id for device_id in device_ids if device_id < 0]
    if invalid_ids:
        raise argparse.ArgumentTypeError("--device-ids must be non-negative integers.")

    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("--device-ids cannot contain duplicates.")

    return device_ids


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run operator benchmarks with unified shape templates."
    )
    selector_group = parser.add_mutually_exclusive_group(required=True)
    selector_group.add_argument(
        "--op",
        help=(
            "Comma-separated operator names to benchmark, for example: "
            "addmm,bmm,conv2d."
        ),
    )
    selector_group.add_argument(
        "--category",
        help="Operator category to benchmark, for example: attention, gemm, or tensor_ops.",
    )
    selector_group.add_argument(
        "--all",
        action="store_true",
        help="Benchmark every registered operator.",
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
        help="Shape family name for family mode, or all to run every family.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when --shape_mode random. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--random-count",
        type=parse_positive_int,
        default=None,
        help="Number of random shape combinations to generate for --shape_mode random. Defaults to operator config or 100.",
    )
    parser.add_argument(
        "--random-max-attempts",
        type=parse_positive_int,
        default=None,
        help="Maximum attempts for generating unique random shapes. Defaults to operator config or 10000.",
    )
    parser.add_argument(
        "--language",
        default="cuda",
        help=(
            "Target backend language, for example cuda, maca, biren, tiangai, "
            "or ascendc."
        ),
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["eager", "compiled", "both"],
        help="Execution mode for PyTorch reference benchmarks.",
    )
    parser.add_argument(
        "--perf-trials",
        type=parse_positive_int,
        default=5,
        help="Number of timing trials per shape case.",
    )
    parser.add_argument(
        "--baseline-compute-mode",
        choices=["correctness", "performance"],
        default="performance",
        help=(
            "Baseline intermediate precision mode. correctness uses FP32 "
            "intermediates; performance keeps the original input dtype."
        ),
    )
    parser.add_argument(
        "--workers",
        type=parse_workers,
        default=1,
        help="Number of concurrent worker processes to launch.",
    )
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=parse_device_ids("0"),
        help="Comma-separated accelerator device ids that workers are allowed to use.",
    )
    return parser


def main() -> None:
    """Parse arguments, run the benchmark, and print unified JSON results."""
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        resolve_operator_selection(
            op_name=args.op,
            category=args.category,
            all_ops=args.all,
        )
    except ValueError as error:
        parser.error(str(error))
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
