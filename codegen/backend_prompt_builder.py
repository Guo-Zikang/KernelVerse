"""Backend-specific prompt context for generated accelerator operators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_BACKEND_PROMPT_PROFILES_PATH = Path(__file__).with_name(
    "backend_prompt_profiles.json"
)
BACKEND_PROMPTS_DIR = Path(__file__).with_name("backend_prompts")
PROFILE_LIST_FIELDS = (
    "implementation_requirements",
    "programming_guidance",
)
PROFILE_FIELDS = {"display_name", *PROFILE_LIST_FIELDS}


def _validate_profile(profile_key: str, profile: Any, profiles_path: Path) -> dict:
    if not isinstance(profile, dict):
        raise ValueError(
            f"Backend prompt profile {profile_key!r} in {profiles_path} must be an object."
        )

    unknown_fields = set(profile) - PROFILE_FIELDS
    if unknown_fields:
        fields = ", ".join(sorted(unknown_fields))
        raise ValueError(
            f"Backend prompt profile {profile_key!r} in {profiles_path} has "
            f"unknown field(s): {fields}."
        )

    display_name = profile.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(
            f"Backend prompt profile {profile_key!r} in {profiles_path} must "
            "define a non-empty display_name string."
        )

    normalized = {"display_name": display_name.strip()}
    for field in PROFILE_LIST_FIELDS:
        values = profile.get(field)
        if not isinstance(values, list):
            raise ValueError(
                f"Backend prompt profile {profile_key!r} field {field!r} in "
                f"{profiles_path} must be a list of strings."
            )
        if any(not isinstance(value, str) for value in values):
            raise ValueError(
                f"Backend prompt profile {profile_key!r} field {field!r} in "
                f"{profiles_path} must contain only strings."
            )
        normalized[field] = [value.strip() for value in values if value.strip()]
    return normalized


def load_backend_prompt_profiles(
    path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load and validate backend prompt profiles."""
    profiles_path = path or DEFAULT_BACKEND_PROMPT_PROFILES_PATH
    try:
        payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Backend prompt profiles file does not exist: {profiles_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid backend prompt profiles JSON in {profiles_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Backend prompt profiles in {profiles_path} must be a JSON object."
        )
    if "_default" not in payload:
        raise ValueError(
            f"Backend prompt profiles in {profiles_path} must define _default."
        )

    profiles = {}
    for profile_key, profile in payload.items():
        if not isinstance(profile_key, str) or not profile_key:
            raise ValueError(
                f"Backend prompt profile keys in {profiles_path} must be non-empty strings."
            )
        profiles[profile_key] = _validate_profile(profile_key, profile, profiles_path)
    return profiles


def resolve_backend_prompt_profile(
    language: str,
    *,
    profiles_path: Optional[Path] = None,
) -> tuple[dict, str, bool]:
    """Resolve an exact backend profile or the generic fallback."""
    profiles = load_backend_prompt_profiles(profiles_path)
    backend_key = str(language).strip().lower()
    if backend_key in profiles and backend_key != "_default":
        return profiles[backend_key], backend_key, False
    return profiles["_default"], "_default", True


def _format_rule_section(title: str, rules: list[str]) -> str:
    if not rules:
        return ""
    return title + ":\n" + "\n".join(f"- {rule}" for rule in rules)


def build_backend_prompt(
    *,
    language: str,
    runtime_env: dict[str, Any],
    profiles_path: Optional[Path] = None,
) -> tuple[str, str, bool]:
    """Build dynamic hardware context and static backend implementation guidance."""
    profile, profile_key, used_fallback = resolve_backend_prompt_profile(
        language,
        profiles_path=profiles_path,
    )
    backend_key = str(language).strip().lower()
    sections = [
        "Target backend:\n"
        f"- Backend key: {backend_key}\n"
        f"- Platform: {profile['display_name']}",
        "Target runtime environment:\n"
        "```json\n"
        f"{json.dumps(runtime_env, indent=2, default=str)}\n"
        "```",
    ]
    section_specs = (
        ("Backend implementation requirements", "implementation_requirements"),
        ("Backend programming guidance", "programming_guidance"),
    )
    for title, field in section_specs:
        section = _format_rule_section(title, profile[field])
        if section:
            sections.append(section)
    standalone_prompt_path = BACKEND_PROMPTS_DIR / f"{backend_key}.txt"
    if standalone_prompt_path.is_file():
        standalone_prompt = standalone_prompt_path.read_text(encoding="utf-8").strip()
        if standalone_prompt:
            sections.append(standalone_prompt)
    return "\n\n".join(sections) + "\n", profile_key, used_fallback
