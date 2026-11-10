"""Helpers for extracting generated Python source from model responses."""

from __future__ import annotations

import re


FENCED_CODE_RE = re.compile(
    r"```(?P<language>[a-zA-Z0-9_+-]*)\s*\n?(?P<code>.*?)```",
    re.DOTALL,
)


def extract_python_code(response_text: str) -> str:
    """Extract the first Python-like fenced code block, or return raw text."""
    text = response_text.strip()
    if not text:
        raise ValueError("Model response was empty.")

    fallback_code = None
    for match in FENCED_CODE_RE.finditer(text):
        language = match.group("language").strip().lower()
        code = match.group("code").strip()
        if language in ("", "python", "py"):
            return code
        if fallback_code is None:
            fallback_code = code

    if fallback_code is not None:
        return fallback_code
    return text


def validate_generated_code(code: str) -> None:
    """Validate the minimum contract expected by the evaluator."""
    if not code.strip():
        raise ValueError("Generated code is empty after extraction.")
    if "class ModelNew" not in code:
        raise ValueError("Generated code must define class ModelNew.")
    compile(code, "<generated_cuda_operator>", "exec")

