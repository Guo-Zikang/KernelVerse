"""Evaluate generated CUDA code against registered reference operators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from operators.registry import get_operator_spec
from runner.generated_code_checker import check_generated_file
from runner.generated_correctness_runner import verify_generated_operator
from runner.random_shape_config import resolve_random_shape_settings


EVALUATION_FIELD_ORDER = [
    "op_name",
    "shape_mode",
    "language",
    "devices",
    "run_id",
    "random",
    "perf_trials_per_shape",
    "generated_code",
    "reward_hacking",
    "compiled",
    "correctness",
    "cases",
]


@dataclass
class EvaluationRequest:
    run_id: str
    op_name: str
    shape_mode: str
    shape_name: str
    shape_family: str
    language: str
    generated_path: Path
    run_dir: Path
    device_ids: List[int]
    verify_trials: int = 5
    perf_trials: int = 5
    random_seed: Optional[int] = 42
    random_count: Optional[int] = None
    random_max_attempts: Optional[int] = None


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluation_success(result: Dict) -> bool:
    correctness = result.get("correctness") or {}
    return bool(correctness.get("success"))


def _order_evaluation_result(result: Dict) -> Dict:
    result.pop("success", None)
    ordered = {key: result[key] for key in EVALUATION_FIELD_ORDER if key in result}
    for key, value in result.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _reward_hacking_summary(reward_hacking_report: Dict) -> Dict:
    return {
        "risk_level": reward_hacking_report.get("risk_level", "low"),
    }


def run_evaluation(request: EvaluationRequest) -> Dict:
    """Run generated-code evaluation and write evaluation.json."""
    reward_hacking_report = check_generated_file(request.generated_path, op_name=request.op_name)
    reward_hacking = _reward_hacking_summary(reward_hacking_report)
    _write_json(request.run_dir / "reward_hacking_report.json", reward_hacking_report)
    try:
        device_id = request.device_ids[0]
        operator_spec = get_operator_spec(request.op_name)
        random_count, random_max_attempts = resolve_random_shape_settings(
            operator_spec,
            random_count=request.random_count,
            random_max_attempts=request.random_max_attempts,
        )
        result = verify_generated_operator(
            op_name=request.op_name,
            language=request.language,
            shape_mode=request.shape_mode,
            shape_name=request.shape_name,
            shape_family=request.shape_family,
            generated_path=str(request.generated_path),
            device_id=device_id,
            trials=request.verify_trials,
            random_seed=request.random_seed,
            random_count=random_count,
            random_max_attempts=random_max_attempts,
            perf_trials=request.perf_trials,
            partial_result_path=str(request.run_dir / "evaluation.json"),
            partial_result_metadata={
                "run_id": request.run_id,
                "reward_hacking": reward_hacking,
            },
        )
        result["run_id"] = request.run_id
        result["reward_hacking"] = reward_hacking
        result = _order_evaluation_result(result)
    except Exception as exc:  # pylint: disable=broad-except
        result = {
            "op_name": request.op_name,
            "shape_mode": request.shape_mode,
            "language": request.language,
            "devices": list(request.device_ids),
            "run_id": request.run_id,
            "perf_trials_per_shape": request.perf_trials,
            "generated_code": str(request.generated_path),
            "reward_hacking": reward_hacking,
            "compiled": None,
            "correctness": {
                "success": False,
                "num_trials": request.verify_trials,
                "failed_trials": [],
                "error_message": str(exc),
            },
            "cases": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        result = _order_evaluation_result(result)

    _write_json(request.run_dir / "evaluation.json", result)
    return result
