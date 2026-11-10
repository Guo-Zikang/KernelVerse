"""Shared prompt and runtime context preparation for code generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from codegen.artifacts import write_text
from codegen.backend_prompt_builder import build_backend_prompt
from codegen.env import load_llm_config, repo_root, require_llm_config
from codegen.model_prompt_adapter import resolve_model_adapter_prompt_for_backend
from codegen.prompt_builder import build_prompt, resolve_selected_shapes
from codegen.prompt_composer import compose_generation_prompt
from codegen.runtime_env import collect_runtime_env
from operators.registry import get_operator_spec
from runner.random_shape_config import resolve_random_shape_settings

if TYPE_CHECKING:
    from codegen.generators import GenerationRequest


@dataclass(frozen=True)
class GenerationContext:
    """Prompt layers and metadata shared by one-shot and agent generators."""

    config: Dict[str, str]
    runtime_env: Dict[str, Any]
    selected_shapes: list[dict]
    random_count: int
    random_max_attempts: int
    benchmark_prompt: str
    backend_prompt: str
    model_adapter_prompt: str
    prompt: str
    benchmark_prompt_path: Path
    backend_prompt_path: Path
    model_adapter_prompt_path: Path
    combined_prompt_path: Path
    backend_profile: str
    backend_profile_fallback: bool
    adapter_model: str | None
    adapter_source: str | None
    adapter_backend: str | None

    def prompt_layers_payload(self) -> dict:
        return {
            "benchmark_prompt": str(self.benchmark_prompt_path),
            "backend_prompt": str(self.backend_prompt_path),
            "model_adapter_prompt": str(self.model_adapter_prompt_path),
            "combined_prompt": str(self.combined_prompt_path),
            "backend_profile": self.backend_profile,
            "backend_profile_fallback": self.backend_profile_fallback,
            "adapter_model": self.adapter_model,
            "adapter_applied": self.adapter_model is not None,
            "adapter_source": self.adapter_source,
            "adapter_backend": self.adapter_backend,
        }


def prepare_generation_context(request: "GenerationRequest") -> GenerationContext:
    """Build and persist the exact prompt layers for one generation task."""
    root = repo_root()
    env_path = request.env_path or root / ".env"
    config = require_llm_config(load_llm_config(env_path))
    artifacts_dir = request.run_dir / "artifacts"

    runtime_env = collect_runtime_env(
        language=request.language,
        device_id=request.device_id,
    )
    operator_spec = get_operator_spec(request.op_name)
    resolved_random_count, resolved_random_max_attempts = resolve_random_shape_settings(
        operator_spec,
        random_count=request.random_count,
        random_max_attempts=request.random_max_attempts,
    )
    selected_shapes = resolve_selected_shapes(
        op_name=request.op_name,
        shape_mode=request.shape_mode,
        shape_name=request.shape_name,
        shape_family=request.shape_family,
        random_count=resolved_random_count,
        random_max_attempts=resolved_random_max_attempts,
        random_seed=request.random_seed,
    )
    benchmark_prompt = build_prompt(
        op_name=request.op_name,
        shape_mode=request.shape_mode,
        shape_name=request.shape_name,
        shape_family=request.shape_family,
        random_count=resolved_random_count,
        random_max_attempts=resolved_random_max_attempts,
        random_seed=request.random_seed,
        selected_shapes=selected_shapes,
        language=request.language,
    )
    backend_prompt, backend_profile, backend_profile_fallback = build_backend_prompt(
        language=request.language,
        runtime_env=runtime_env,
    )
    (
        model_adapter_prompt,
        adapter_model,
        adapter_source,
        adapter_backend,
    ) = resolve_model_adapter_prompt_for_backend(
        config["LLM_MODEL"],
        request.language,
    )
    prompt = compose_generation_prompt(
        benchmark_prompt,
        backend_prompt,
        model_adapter_prompt,
    )

    benchmark_prompt_path = artifacts_dir / "benchmark_prompt.txt"
    backend_prompt_path = artifacts_dir / "backend_prompt.txt"
    model_adapter_prompt_path = artifacts_dir / "model_adapter_prompt.txt"
    combined_prompt_path = artifacts_dir / "prompt.txt"
    write_text(benchmark_prompt_path, benchmark_prompt)
    write_text(backend_prompt_path, backend_prompt)
    write_text(model_adapter_prompt_path, model_adapter_prompt)
    write_text(combined_prompt_path, prompt)

    return GenerationContext(
        config=config,
        runtime_env=runtime_env,
        selected_shapes=selected_shapes,
        random_count=resolved_random_count,
        random_max_attempts=resolved_random_max_attempts,
        benchmark_prompt=benchmark_prompt,
        backend_prompt=backend_prompt,
        model_adapter_prompt=model_adapter_prompt,
        prompt=prompt,
        benchmark_prompt_path=benchmark_prompt_path,
        backend_prompt_path=backend_prompt_path,
        model_adapter_prompt_path=model_adapter_prompt_path,
        combined_prompt_path=combined_prompt_path,
        backend_profile=backend_profile,
        backend_profile_fallback=backend_profile_fallback,
        adapter_model=adapter_model,
        adapter_source=adapter_source,
        adapter_backend=adapter_backend,
    )
