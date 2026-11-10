"""Small .env parser for code generation configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def repo_root() -> Path:
    """Return the op_benchmark root directory."""
    return Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> Dict[str, str]:
    """Load key/value pairs from a .env file without third-party dependencies."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env line {line_number}: empty key.")
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            value = value[1:-1]
        values[key] = value
    return values


def load_llm_config(env_path: Path | None = None) -> Dict[str, str]:
    """Load LLM config from process env overlaid on op_benchmark/.env."""
    resolved_env_path = env_path or repo_root() / ".env"
    file_values = load_env_file(resolved_env_path)
    config = dict(file_values)
    for key in (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
    ):
        if os.environ.get(key) is not None:
            config[key] = os.environ[key]
    return config


def require_llm_config(config: Dict[str, str]) -> Dict[str, str]:
    """Validate and normalize required LLM config fields."""
    missing = [key for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not config.get(key)]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing LLM config value(s): {missing_text}. "
            "Set them in op_benchmark/.env or the process environment."
        )

    normalized = dict(config)
    normalized.setdefault("LLM_TEMPERATURE", "0.2")
    normalized.setdefault("LLM_MAX_TOKENS", "12000")
    return normalized

