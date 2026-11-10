"""Subprocess worker for one generated-code compile or case evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operators.registry import get_operator_spec, load_object
from runner.correctness_policy import resolve_correctness_policy
from runner.generated_correctness_runner import (
    _case_subprocess_failure_result,
    _compile_generated_code,
    _operator_tolerance_policy,
    _resolve_correctness_baseline_compute_mode,
    _run_generated_case,
)
from runner.profiler_runner import ensure_backend, setup_backend_device


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _device_name(backend: Any, device: Any) -> str:
    return backend.get_hardware_name() if hasattr(backend, "get_hardware_name") else str(device)


def _run_compile_request(request: Dict[str, Any]) -> Dict[str, Any]:
    backend = ensure_backend(str(request["language"]))
    device = setup_backend_device(backend, str(request["language"]), int(request["device_id"]))
    try:
        generated_path = Path(str(request["generated_path"]))
        compiled, compile_info = _compile_generated_code(
            backend,
            generated_path.read_text(encoding="utf-8"),
            str(request["op_name"]),
        )
        return {
            "mode": "compile",
            "device": _device_name(backend, device),
            "compiled": bool(compiled),
            "compile_info": compile_info,
            "has_model_new": backend.context.get("ModelNew") is not None,
        }
    finally:
        if hasattr(backend, "cleanup"):
            try:
                backend.cleanup()
            except Exception:
                pass


def _run_case_request(request: Dict[str, Any]) -> Dict[str, Any]:
    op_name = str(request["op_name"])
    language = str(request["language"])
    device_id = int(request["device_id"])
    case_index = int(request["case_index"])
    raw_shape = dict(request["raw_shape"])

    backend = ensure_backend(language)
    device = setup_backend_device(backend, language, device_id)
    try:
        operator_spec = get_operator_spec(op_name)
        generated_path = Path(str(request["generated_path"]))
        compiled, compile_info = _compile_generated_code(
            backend,
            generated_path.read_text(encoding="utf-8"),
            op_name,
        )
        if not compiled:
            return _case_subprocess_failure_result(
                raw_shape=raw_shape,
                case_index=case_index,
                error_type="CaseCompileFailed",
                error_message=str(compile_info or "Generated code compilation failed."),
                device_id=device_id,
            )

        generated_model_class = backend.context.get("ModelNew")
        if generated_model_class is None:
            return _case_subprocess_failure_result(
                raw_shape=raw_shape,
                case_index=case_index,
                error_type="ModelNewMissing",
                error_message="Generated code must define ModelNew.",
                device_id=device_id,
            )

        reference_model_class = load_object(operator_spec["model_class_path"])
        generate_inputs_fn = load_object(operator_spec["generate_inputs_path"])
        tolerance_policy = _operator_tolerance_policy(operator_spec)
        correctness_policy = resolve_correctness_policy(operator_spec)
        correctness_baseline_compute_mode = str(
            request.get("correctness_baseline_compute_mode")
            or _resolve_correctness_baseline_compute_mode(operator_spec)
        )

        return _run_generated_case(
            op_name=op_name,
            language=language,
            device_id=device_id,
            case_index=case_index,
            raw_shape=raw_shape,
            generated_model_class=generated_model_class,
            reference_model_class=reference_model_class,
            generate_inputs_fn=generate_inputs_fn,
            backend=backend,
            device=device,
            trials=int(request["trials"]),
            perf_trials=int(request["perf_trials"]),
            tolerance_policy=tolerance_policy,
            correctness_policy=correctness_policy,
            correctness_baseline_compute_mode=correctness_baseline_compute_mode,
        )
    finally:
        if hasattr(backend, "cleanup"):
            try:
                backend.cleanup()
            except Exception:
                pass


def _error_result(request: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_message = str(error)
    if request.get("mode") == "case":
        return _case_subprocess_failure_result(
            raw_shape=dict(request.get("raw_shape") or {}),
            case_index=int(request.get("case_index", -1)),
            error_type=error_type,
            error_message=error_message,
            device_id=int(request.get("device_id", -1)),
        )
    return {
        "mode": request.get("mode"),
        "device": None,
        "compiled": False,
        "compile_info": error_message,
        "has_model_new": False,
        "error_type": error_type,
        "error_message": error_message,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one generated-code case in a subprocess.")
    parser.add_argument("--request-json", required=True, help="Path to the case request JSON.")
    parser.add_argument("--result-json", required=True, help="Path where the result JSON is written.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request_path = Path(args.request_json)
    result_path = Path(args.result_json)
    request: Dict[str, Any] = {}
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        mode = request.get("mode")
        if mode == "compile":
            result = _run_compile_request(request)
        elif mode == "case":
            result = _run_case_request(request)
        else:
            raise ValueError(f"Unsupported generated case worker mode: {mode!r}.")
    except Exception as exc:  # pylint: disable=broad-except
        result = _error_result(request, exc)

    _write_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
