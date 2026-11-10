"""Shared run-directory path helpers."""

from __future__ import annotations

from pathlib import Path

from operators.registry import get_operator_spec


def op_run_dir(run_dir: Path, op_name: str) -> Path:
    """Return the category-grouped run directory for one operator."""
    category = get_operator_spec(op_name)["category"]
    return run_dir / "ops" / category / op_name


def legacy_op_run_dir(run_dir: Path, op_name: str) -> Path:
    """Return the pre-category-layout run directory for one operator."""
    return run_dir / "ops" / op_name


def existing_op_run_dir(run_dir: Path, op_name: str) -> Path:
    """Prefer the category-grouped path, then fall back to the legacy path."""
    grouped_dir = op_run_dir(run_dir, op_name)
    if grouped_dir.exists():
        return grouped_dir
    legacy_dir = legacy_op_run_dir(run_dir, op_name)
    if legacy_dir.exists():
        return legacy_dir
    return grouped_dir
