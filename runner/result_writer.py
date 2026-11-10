"""Utilities for writing benchmark result JSON files."""

import json
import os
from datetime import datetime
from typing import Dict, Optional


def write_result_json(
    base_dir: str,
    op_name: str,
    mode: str,
    result: Dict,
    execution_mode: Optional[str] = None,
) -> str:
    """Write benchmark results into the results directory."""
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [op_name, mode]
    if execution_mode:
        parts.append(execution_mode)
    file_name = f"{'_'.join(parts)}_{timestamp}.json"
    output_path = os.path.join(results_dir, file_name)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)

    return output_path
