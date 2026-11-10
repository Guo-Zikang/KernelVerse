"""Operator package discovery and loading."""

from __future__ import annotations

import importlib.util
import importlib
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional


REQUIRED_PACKAGE_FILES = ("shapes.py", "config.json", "docs.md", "helpers.py")


def repo_root() -> Path:
    """Return the op_benchmark root directory."""
    return Path(__file__).resolve().parents[1]


def operators_root() -> Path:
    """Return the root directory containing operator categories."""
    return Path(__file__).resolve().parent


def _is_category_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"__pycache__", "Attention_ref", "single_github"}
    )


def _operator_file(op_dir: Path) -> Path:
    return op_dir / f"{op_dir.name}.py"


def _is_operator_package(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    if not _operator_file(path).exists():
        return False
    return all((path / file_name).exists() for file_name in REQUIRED_PACKAGE_FILES)


def _load_config(config_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid operator config JSON: {config_path}") from exc


def _build_spec(category: str, op_dir: Path) -> Dict[str, Any]:
    op_name = op_dir.name
    operator_file = _operator_file(op_dir)
    shapes_file = op_dir / "shapes.py"
    config_path = op_dir / "config.json"
    docs_path = op_dir / "docs.md"
    helpers_path = op_dir / "helpers.py"
    module_name = f"operators.{category}.{op_name}.{op_name}"
    return {
        "op_name": op_name,
        "category": category,
        "op_dir": str(op_dir),
        "operator_file": str(operator_file),
        "shapes_file": str(shapes_file),
        "config_path": str(config_path),
        "docs_path": str(docs_path),
        "helpers_path": str(helpers_path),
        "module_name": module_name,
        "model_class_path": f"{operator_file}::Model",
        "generate_inputs_path": f"{operator_file}::generate_inputs",
        "shape_module": str(shapes_file),
        "config": _load_config(config_path),
    }


@lru_cache(maxsize=1)
def discover_operator_specs() -> Dict[str, Dict[str, Any]]:
    """Discover operator packages under operators/<category>/<op_name>/."""
    discovered: Dict[str, Dict[str, Any]] = {}
    duplicates: Dict[str, List[str]] = {}

    for category_dir in sorted(path for path in operators_root().iterdir() if _is_category_dir(path)):
        for op_dir in sorted(category_dir.iterdir()):
            if not _is_operator_package(op_dir):
                continue
            op_name = op_dir.name
            if op_name in discovered:
                duplicates.setdefault(op_name, [discovered[op_name]["op_dir"]]).append(str(op_dir))
                continue
            discovered[op_name] = _build_spec(category_dir.name, op_dir)

    if duplicates:
        details = "; ".join(
            f"{op_name}: {', '.join(paths)}" for op_name, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate operator package name(s): {details}")
    return discovered


def get_operator_spec(op_name: str) -> Dict[str, Any]:
    """Return the discovered spec for one operator."""
    specs = discover_operator_specs()
    if op_name not in specs:
        raise ValueError(
            f"Invalid operator name: {op_name}. Expected an operator package at "
            f"operators/<category>/{op_name}/{op_name}.py with shapes.py, config.json, docs.md, and helpers.py."
        )
    return specs[op_name]


def list_operator_names() -> List[str]:
    """Return all discovered operator names."""
    return list(discover_operator_specs().keys())


def list_categories() -> List[str]:
    """Return all discovered operator categories."""
    categories = {spec["category"] for spec in discover_operator_specs().values()}
    return sorted(categories)


def list_operator_names_by_category(category: str) -> List[str]:
    """Return discovered operator names for one category in registry order."""
    specs = discover_operator_specs()
    categories = list_categories()
    if category not in categories:
        raise ValueError(
            f"Invalid operator category: {category}. Supported categories: {', '.join(categories)}"
        )
    return [
        op_name
        for op_name, spec in specs.items()
        if spec["category"] == category
    ]


def parse_operator_names(value: str) -> List[str]:
    """Parse and validate a comma-separated operator selector."""
    op_names = [item.strip() for item in str(value).split(",")]
    if not op_names or any(not op_name for op_name in op_names):
        raise ValueError("--op contains an empty operator name.")
    if len(op_names) != len(set(op_names)):
        raise ValueError("--op cannot contain duplicate operator names.")
    for op_name in op_names:
        get_operator_spec(op_name)
    return op_names


def resolve_operator_selection(
    *,
    op_name: Optional[str] = None,
    category: Optional[str] = None,
    all_ops: bool = False,
) -> List[str]:
    """Resolve exactly one operator selector into operator names."""
    selectors = [bool(op_name), bool(category), bool(all_ops)]
    if sum(selectors) != 1:
        raise ValueError("Specify exactly one of --op, --category, or --all.")

    if op_name:
        return parse_operator_names(op_name)

    if category:
        return list_operator_names_by_category(category)

    return list_operator_names()


@lru_cache(maxsize=None)
def load_module(module_ref: str) -> ModuleType:
    """Load a module by file path."""
    path = Path(module_ref)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise ValueError(f"Module file not found: {path}")

    try:
        relative_path = path.relative_to(repo_root()).with_suffix("")
        module_name = ".".join(relative_path.parts)
        return importlib.import_module(module_name)
    except ValueError:
        pass

    module_name = "op_benchmark_dynamic_" + "_".join(path.with_suffix("").parts[-4:])
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load module from file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_object(object_ref: str) -> Any:
    """Load an object from a file reference of the form /path/file.py::Name."""
    if "::" not in object_ref:
        raise ValueError(f"Invalid object reference: {object_ref}. Expected file.py::ObjectName.")
    module_ref, object_name = object_ref.rsplit("::", 1)
    module = load_module(module_ref)
    return getattr(module, object_name)
