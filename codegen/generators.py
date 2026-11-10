"""Generator abstractions for future LLM or agent implementations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

from codegen.artifacts import write_json, write_text
from codegen.code_extractor import extract_python_code, validate_generated_code
from codegen.generation_context import prepare_generation_context
from codegen.llm_client import OpenAICompatibleClient
from codegen.runtime_env import collect_runtime_env
from runner.generated_code_checker import check_generated_code


MODELS_WITHOUT_TEMPERATURE = {
    "claude-4.7-opus",
}


@dataclass
class GenerationRequest:
    run_id: str
    op_name: str
    shape_mode: str
    shape_name: str
    shape_family: str
    language: str
    run_dir: Path
    device_id: int = 0
    env_path: Optional[Path] = None
    no_overwrite: bool = False
    random_count: Optional[int] = None
    random_max_attempts: Optional[int] = None
    random_seed: Optional[int] = 42
    candidate_id: Optional[int] = None
    samples_per_op: int = 1
    temperature: Optional[float] = None


class CodeGenerator(Protocol):
    """Interface implemented by LLM and future agent generators."""

    def generate(self, request: GenerationRequest) -> Dict:
        """Generate code and return a generation result payload."""


GenerationRetryFn = Callable[[int, int, Dict], None]


def generation_result_context(request: GenerationRequest) -> Dict:
    context = {
        "run_id": request.run_id,
        "op_name": request.op_name,
        "shape_mode": request.shape_mode,
        "language": request.language,
    }
    if request.candidate_id is not None:
        context["candidate_id"] = request.candidate_id
        context["samples_per_op"] = request.samples_per_op
    if request.shape_mode == "single":
        context["shape_name"] = request.shape_name
    elif request.shape_mode == "family":
        context["shape_family"] = request.shape_family
    return context


class LLMCodeGenerator:
    """Single-call OpenAI-compatible generator."""

    generator_type = "llm"

    @staticmethod
    def _resolve_temperature(
        *,
        model: str,
        requested_temperature: Optional[float],
        default_temperature: str,
    ) -> Optional[float]:
        if model in MODELS_WITHOUT_TEMPERATURE:
            return None
        if requested_temperature is not None:
            return float(requested_temperature)
        return float(default_temperature)

    def generate(self, request: GenerationRequest) -> Dict:
        generated_path = request.run_dir / "generated_ops" / f"{request.op_name}.txt"
        artifacts_dir = request.run_dir / "artifacts"
        if generated_path.exists() and request.no_overwrite:
            raise FileExistsError(f"Generated code file already exists: {generated_path}")
        context = prepare_generation_context(request)
        config = context.config
        model = config["LLM_MODEL"]
        temperature = self._resolve_temperature(
            model=model,
            requested_temperature=request.temperature,
            default_temperature=config["LLM_TEMPERATURE"],
        )
        client = OpenAICompatibleClient(
            base_url=config["LLM_BASE_URL"],
            api_key=config["LLM_API_KEY"],
            model=model,
            temperature=temperature,
            max_tokens=int(config["LLM_MAX_TOKENS"]),
        )
        response_text = client.generate(context.prompt)
        write_text(artifacts_dir / "response.txt", response_text)

        code = extract_python_code(response_text)
        validate_generated_code(code)
        reward_hacking_report = check_generated_code(code, op_name=request.op_name)
        write_json(request.run_dir / "reward_hacking_report.json", reward_hacking_report)
        write_text(generated_path, code)
        write_text(artifacts_dir / "code.py", code)

        result = {
            **generation_result_context(request),
            "success": True,
            "runtime_env": context.runtime_env,
            "generator": {
                "type": self.generator_type,
                "model": model,
                "base_url": config["LLM_BASE_URL"],
                "temperature": temperature,
            },
            "random_shape": {
                "seed": request.random_seed,
                "count": context.random_count,
                "max_attempts": context.random_max_attempts,
            },
            "selected_shapes": context.selected_shapes,
            "reward_hacking": {
                "risk_level": reward_hacking_report.get("risk_level", "low"),
            },
            "prompt_layers": context.prompt_layers_payload(),
            "outputs": {
                "generated_code": str(generated_path),
                "prompt": str(context.combined_prompt_path),
                "response": str(artifacts_dir / "response.txt"),
                "code": str(artifacts_dir / "code.py"),
            },
        }
        write_json(request.run_dir / "generation.json", result)
        return result


def run_generation(request: GenerationRequest, generator: Optional[CodeGenerator] = None) -> Dict:
    """Run code generation and always write generation.json."""
    selected_generator = generator or LLMCodeGenerator()
    try:
        return selected_generator.generate(request)
    except Exception as exc:  # pylint: disable=broad-except
        result = {
            **generation_result_context(request),
            "success": False,
            "runtime_env": collect_runtime_env(
                language=request.language,
                device_id=request.device_id,
            ),
            "generator": {
                "type": getattr(selected_generator, "generator_type", type(selected_generator).__name__),
            },
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        write_json(request.run_dir / "generation.json", result)
        return result


def run_generation_with_retries(
    request: GenerationRequest,
    *,
    generation_retries: int,
    generator: Optional[CodeGenerator] = None,
    initial_result: Optional[Dict] = None,
    on_retry: Optional[GenerationRetryFn] = None,
) -> Dict:
    """Run generation and retry failed results in the current worker."""
    if generation_retries < 0:
        raise ValueError("generation_retries must be >= 0.")

    result = initial_result
    retries_used = 0
    total_attempts = 1 if initial_result is not None else 0
    retry_request = request

    if result is None:
        result = run_generation(retry_request, generator=generator)
        total_attempts = 1

    while not result.get("success") and retries_used < generation_retries:
        retries_used += 1
        if on_retry is not None:
            on_retry(retries_used, generation_retries, result)
        retry_request = replace(request, no_overwrite=False)
        result = run_generation(retry_request, generator=generator)
        total_attempts += 1

    result["generation_retry"] = {
        "configured_retries": generation_retries,
        "retries_used": retries_used,
        "total_attempts": total_attempts,
    }
    write_json(request.run_dir / "generation.json", result)
    return result
