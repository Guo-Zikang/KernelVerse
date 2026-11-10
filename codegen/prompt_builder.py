"""Platform-neutral prompt construction for generated accelerator operators."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Optional

from operators.registry import get_operator_spec, load_object
from runner.benchmark_runner import resolve_shapes
from runner.random_shape_config import resolve_random_shape_settings


OPERATOR_PROMPT_TEMPLATES_DIR = Path(__file__).with_name("operator_prompt_templates")


def _extract_reference_source(operator_path: Path) -> str:
    """Extract only Model and generate_inputs from the reference operator file."""
    source = operator_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            segment = ast.get_source_segment(source, node)
            if segment:
                segments.append(segment)
        elif isinstance(node, ast.FunctionDef) and node.name == "generate_inputs":
            segment = ast.get_source_segment(source, node)
            if segment:
                segments.append(segment)

    missing = []
    if not any(segment.startswith("class Model") for segment in segments):
        missing.append("class Model")
    if not any(segment.startswith("def generate_inputs") for segment in segments):
        missing.append("generate_inputs")
    if missing:
        raise ValueError(
            f"Reference operator file {operator_path} is missing: {', '.join(missing)}"
        )
    return "\n\n\n".join(segments)


def resolve_selected_shapes(
    *,
    op_name: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_count: Optional[int],
    random_max_attempts: Optional[int],
    random_seed: Optional[int] = 42,
) -> list[Dict[str, Any]]:
    """Resolve the exact benchmark shapes used in the generation prompt."""
    operator_spec = get_operator_spec(op_name)
    validate_shape_fn = load_object(f"{operator_spec['helpers_path']}::validate_shape")
    resolved_random_count, resolved_random_max_attempts = resolve_random_shape_settings(
        operator_spec,
        random_count=random_count,
        random_max_attempts=random_max_attempts,
    )
    _, shapes, _ = resolve_shapes(
        operator_spec["shape_module"],
        shape_mode,
        shape_name,
        shape_family,
        random_count=resolved_random_count,
        random_max_attempts=resolved_random_max_attempts,
        random_seed=random_seed,
        validate_shape_fn=validate_shape_fn,
    )
    return shapes


def _format_shape_value(value: Any) -> str:
    """Format one shape table cell with compact structured values."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), default=str)
    return str(value)


def _format_shapes_table(shapes: list[Dict[str, Any]]) -> str:
    """Format selected shapes as a compact prompt-friendly table."""
    if not shapes:
        return "(none)"

    keys = list(shapes[0].keys())
    for shape in shapes[1:]:
        for key in shape:
            if key not in keys:
                keys.append(key)

    rows = [" | ".join(keys)]
    for shape in shapes:
        rows.append(" | ".join(_format_shape_value(shape.get(key)) for key in keys))
    return "\n".join(rows)


def _build_ascendc_prompt(
    *,
    op_name: str,
    operator_source: str,
    selected_shapes: list[Dict[str, Any]],
    docs_source: str,
) -> str:
    template_path = OPERATOR_PROMPT_TEMPLATES_DIR / "ascendc.txt"
    try:
        prompt = template_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Ascend C prompt template does not exist: {template_path}") from exc

    replacements = {
        "__OP_NAME__": op_name,
        "__REFERENCE_SOURCE__": operator_source,
        "__SELECTED_SHAPES__": _format_shapes_table(selected_shapes),
        "__OPERATOR_DOCS__": docs_source,
    }
    for placeholder, value in replacements.items():
        if placeholder not in prompt:
            raise ValueError(
                f"Ascend C prompt template {template_path} is missing {placeholder}."
            )
        prompt = prompt.replace(placeholder, value)
    return prompt.rstrip() + "\n"


def build_prompt(
    *,
    op_name: str,
    shape_mode: str,
    shape_name: str,
    shape_family: str,
    random_count: Optional[int] = None,
    random_max_attempts: Optional[int] = None,
    random_seed: Optional[int] = 42,
    selected_shapes: Optional[list[Dict[str, Any]]] = None,
    language: str = "cuda",
) -> str:
    """Build a prompt with all context needed to generate ModelNew."""
    operator_spec = get_operator_spec(op_name)
    operator_source = _extract_reference_source(Path(operator_spec["operator_file"]))
    docs_source = Path(operator_spec["docs_path"]).read_text(encoding="utf-8")
    if selected_shapes is None:
        selected_shapes = resolve_selected_shapes(
            op_name=op_name,
            shape_mode=shape_mode,
            shape_name=shape_name,
            shape_family=shape_family,
            random_count=random_count,
            random_max_attempts=random_max_attempts,
            random_seed=random_seed,
        )

    if str(language).strip().lower() == "ascendc":
        return _build_ascendc_prompt(
            op_name=op_name,
            operator_source=operator_source,
            selected_shapes=selected_shapes,
            docs_source=docs_source,
        )

    return f"""You are an expert GPU kernel engineer.
Your task is to write a complete, runnable custom operator implementation for a given PyTorch op.

Generate one Python source file for operator: {op_name}
Output code only. Do not use markdown fences or explanations.

Required structure:
- imports
- class ModelNew(nn.Module)
- ModelNew.__init__(self, shape)
- ModelNew.forward(...) with the same signature and output structure as the reference Model.forward(...)

Hard contract:
- ModelNew.forward(...) must not call PyTorch functions for the core computation.
- ModelNew.forward(...) must not call existing operator libraries such as cuBLAS, cuDNN, CUTLASS, or vendor BLAS/DNN libraries.
- ModelNew.forward(...) must not move the core computation to CPU.
- The core computation must be implemented by custom target-backend kernels using the implementation contract in the backend context.
- ModelNew.forward() should dispatch directly to the target-backend implementation and avoid extra Python-side work.
- Do not reproduce reference-only `_baseline_compute_mode` branches; ModelNew must implement dtype-specific target-backend paths based on tensor dtype or `shape["dtype"]`.
- If the reference Model creates trainable parameters in __init__, ModelNew.__init__ must create the same parameters in the same order so benchmark seeding initializes identical weights.
- Do not recreate reference helper functions such as _resolve_dtype or validate_shape.
- Use APIs compatible with the target PyTorch and accelerator runtime versions provided in the backend context.

Optimization goals:
- Correctness has priority over speed.
- Among correct implementations, minimize operator runtime.

Optimization guidance:
- Useful strategies include tiling, shared memory reuse, coalesced memory access, vectorized loads/stores, suitable block sizes, low branch divergence, and careful accumulation dtype.
- Avoid unnecessary tensor allocations, dtype conversions, synchronization, and Python-side overhead.
- For FP16/BF16 inputs, keep most computation in low precision for speed, but promote numerically sensitive steps to FP32 when needed, such as reductions, accumulation, softmax max/sum/logsumexp, normalization variance/RMS computation, and long dot-product accumulations.
- Use hardware matrix acceleration for FP16/BF16 only when supported by the target backend; do not assume NVIDIA-specific WMMA/MMA support on non-NVIDIA backends.
- Choose features and kernel code supported by the target backend and architecture described in the backend context.

Reference implementation excerpt:
```python
{operator_source}
```

Selected benchmark shapes:
```text
{_format_shapes_table(selected_shapes)}
```

Operator notes:
```markdown
{docs_source}
```
"""
