"""Model-specific prompt adapters for code generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_PROMPT_PROFILES_PATH = Path(__file__).with_name(
    "model_prompt_profiles.json"
)
DEFAULT_BACKEND_MODEL_PROMPT_PROFILES_PATH = Path(__file__).with_name(
    "backend_model_prompt_profiles.json"
)


def _normalize_instructions(
    instructions: object,
    *,
    profile_label: str,
    profiles_path: Path,
) -> list[str]:
    if not isinstance(instructions, list):
        raise ValueError(
            f"Model prompt profile for {profile_label} in {profiles_path} "
            "must be a list of strings."
        )
    if any(not isinstance(instruction, str) for instruction in instructions):
        raise ValueError(
            f"Model prompt profile for {profile_label} in {profiles_path} "
            "must contain only strings."
        )
    return [instruction.strip() for instruction in instructions if instruction.strip()]


def _read_profiles_json(profiles_path: Path, *, profile_kind: str) -> object:
    try:
        return json.loads(profiles_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"{profile_kind} prompt profiles file does not exist: {profiles_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid {profile_kind.lower()} prompt profiles JSON in "
            f"{profiles_path}: {exc}"
        ) from exc


def load_model_prompt_profiles(
    path: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Load exact model-name to model-specific instruction lists."""
    profiles_path = path or DEFAULT_MODEL_PROMPT_PROFILES_PATH
    payload = _read_profiles_json(profiles_path, profile_kind="Model")

    if not isinstance(payload, dict):
        raise ValueError(
            f"Model prompt profiles in {profiles_path} must be a JSON object."
        )

    profiles: dict[str, list[str]] = {}
    for model_name, instructions in payload.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                f"Model prompt profile keys in {profiles_path} must be non-empty strings."
            )
        profiles[model_name] = _normalize_instructions(
            instructions,
            profile_label=repr(model_name),
            profiles_path=profiles_path,
        )
    return profiles


def load_backend_model_prompt_profiles(
    path: Optional[Path] = None,
) -> dict[str, dict[str, list[str]]]:
    """Load exact backend/model adapter overrides."""
    profiles_path = path or DEFAULT_BACKEND_MODEL_PROMPT_PROFILES_PATH
    payload = _read_profiles_json(profiles_path, profile_kind="Backend-model")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Backend-model prompt profiles in {profiles_path} must be a JSON object."
        )

    profiles: dict[str, dict[str, list[str]]] = {}
    for backend_name, model_profiles in payload.items():
        if not isinstance(backend_name, str) or not backend_name.strip():
            raise ValueError(
                f"Backend-model prompt profile keys in {profiles_path} must be "
                "non-empty strings."
            )
        if not isinstance(model_profiles, dict):
            raise ValueError(
                f"Backend-model profile {backend_name!r} in {profiles_path} "
                "must be an object."
            )
        normalized_backend = backend_name.strip().lower()
        profiles[normalized_backend] = {}
        for model_name, instructions in model_profiles.items():
            if not isinstance(model_name, str) or not model_name:
                raise ValueError(
                    f"Model keys for backend {backend_name!r} in {profiles_path} "
                    "must be non-empty strings."
                )
            profiles[normalized_backend][model_name] = _normalize_instructions(
                instructions,
                profile_label=f"{backend_name!r}/{model_name!r}",
                profiles_path=profiles_path,
            )
    return profiles


def _format_adapter_prompt(instructions: list[str]) -> str:
    if not instructions:
        return ""
    return "Model-specific instructions:\n" + "\n".join(
        f"- {instruction}" for instruction in instructions
    )


def resolve_model_adapter_prompt(
    model_name: str,
    *,
    profiles_path: Optional[Path] = None,
) -> tuple[str, Optional[str]]:
    """Resolve an adapter by exact model name, returning empty for unknown models."""
    instructions = load_model_prompt_profiles(profiles_path).get(model_name, [])
    if not instructions:
        return "", None
    return _format_adapter_prompt(instructions), model_name


def resolve_model_adapter_prompt_for_backend(
    model_name: str,
    language: str,
    *,
    profiles_path: Optional[Path] = None,
    backend_profiles_path: Optional[Path] = None,
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Resolve a backend/model override before the ordinary model adapter."""
    backend_name = str(language).strip().lower()
    backend_profiles = load_backend_model_prompt_profiles(backend_profiles_path)
    backend_instructions = backend_profiles.get(backend_name, {}).get(model_name)
    if backend_instructions is not None:
        prompt = _format_adapter_prompt(backend_instructions)
        return prompt, model_name if prompt else None, "backend_model", backend_name

    prompt, adapter_model = resolve_model_adapter_prompt(
        model_name,
        profiles_path=profiles_path,
    )
    if adapter_model is None:
        return "", None, None, None
    return prompt, adapter_model, "model", None
