"""Biren backend for native BIRENSUPA generated operator code."""

from __future__ import annotations

import ast
import os
import re
import time

import torch

from backends.backend_registry import Backend, register_backend
from utils.biren_environment import prepare_biren_environment


LEGACY_CUDA_HEADERS = {
    "cuda.h",
    "cuda_bf16.h",
    "cuda_fp16.h",
    "cuda_runtime.h",
    "cuda_runtime_api.h",
}
LEGACY_CUDA_HEADER_PREFIXES = ("ATen/cuda/", "c10/cuda/")

LEGACY_CUDA_KEYWORDS = {"cuda_sources", "extra_cuda_cflags", "with_cuda"}
CPP_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
CPP_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
CPP_INCLUDE_RE = re.compile(
    r"^\s*#\s*include\s*[<\"](?P<header>[^>\"]+)[>\"]",
    re.MULTILINE,
)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _legacy_cuda_headers(source: str) -> set[str]:
    """Return actual CUDA include directives, ignoring comments and identifiers."""
    uncommented = CPP_BLOCK_COMMENT_RE.sub("", source)
    uncommented = CPP_LINE_COMMENT_RE.sub("", uncommented)
    headers = {
        match.group("header").strip()
        for match in CPP_INCLUDE_RE.finditer(uncommented)
    }
    return {
        header
        for header in headers
        if header in LEGACY_CUDA_HEADERS
        or header.startswith(LEGACY_CUDA_HEADER_PREFIXES)
    }


def _ensure_torch_supa(timeout_seconds: float = 10.0):
    """Import torch_br and wait until it has registered torch.supa."""
    prepare_biren_environment()
    try:
        import torch_br  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while not hasattr(torch, "supa"):
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.001)
    return torch_br


@register_backend("biren")
class BirenBackend(Backend):
    """Compile and evaluate native SUPA kernels with torch_br and BRCC."""

    language = "biren"

    def __init__(self):
        self.context = {}
        self.device = None
        self._torch_br = None

    def _require_runtime(self):
        if self._torch_br is None or not hasattr(torch, "supa"):
            self._torch_br = _ensure_torch_supa()
        if self._torch_br is None or not hasattr(torch, "supa"):
            raise RuntimeError(
                "Biren runtime is unavailable. Install torch_br and activate "
                "the BIRENSUPA SDK."
            )
        return self._torch_br

    def get_device(self):
        return self.device

    def set_device(self, device_id: int):
        self._require_runtime()
        self.device = torch.device(f"supa:{device_id}")
        torch.supa.set_device(device_id)

    def setup_device(self, device_id: int):
        """Expose one physical GPU and bind this worker to visible device zero."""
        os.environ["SUPA_VISIBLE_DEVICES"] = str(device_id)
        self._require_runtime()
        if not torch.supa.is_available():
            raise RuntimeError("Biren SUPA device is unavailable.")
        visible_device_count = int(torch.supa.device_count())
        if visible_device_count != 1:
            raise RuntimeError(
                "Biren worker device isolation failed: expected exactly one visible "
                f"SUPA device after setting SUPA_VISIBLE_DEVICES={device_id}, but "
                f"found {visible_device_count}. Set SUPA_VISIBLE_DEVICES before "
                "importing torch_br."
            )
        torch.supa.set_device(0)
        self.device = torch.device("supa:0")
        return self.device

    def get_hardware_name(self):
        configured_name = os.environ.get("BIREN_DEVICE_NAME")
        if configured_name:
            return configured_name
        self._require_runtime()
        device = self.device or torch.device("supa:0")
        return torch.supa.get_device_name(device=device)

    @staticmethod
    def _validate_native_supa_code(code: str) -> None:
        tree = ast.parse(code, filename="<generated_biren_operator>")
        imports_torch_br = any(
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "torch_br" or alias.name.startswith("torch_br.")
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == "torch_br"
                    or str(node.module).startswith("torch_br.")
                )
            )
            for node in ast.walk(tree)
        )
        model_new_found = any(
            isinstance(node, ast.ClassDef) and node.name == "ModelNew"
            for node in ast.walk(tree)
        )

        load_inline_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _dotted_name(node.func).split(".")[-1] == "load_inline"
        ]
        load_inline_keywords = {
            keyword.arg
            for call in load_inline_calls
            for keyword in call.keywords
            if keyword.arg is not None
        }

        legacy = sorted(LEGACY_CUDA_KEYWORDS & load_inline_keywords)
        for node in ast.walk(tree):
            dotted_name = _dotted_name(node)
            if dotted_name == "torch.cuda" or dotted_name.startswith("torch.cuda."):
                legacy.append("torch.cuda")
            if dotted_name == "torch.utils.cpp_extension" or dotted_name.startswith(
                "torch.utils.cpp_extension."
            ):
                legacy.append("torch.utils.cpp_extension")
            if isinstance(node, ast.ImportFrom) and (
                node.module == "torch.utils.cpp_extension"
                or str(node.module).startswith("torch.utils.cpp_extension.")
            ):
                legacy.append("torch.utils.cpp_extension")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                legacy.extend(_legacy_cuda_headers(node.value))

        legacy = sorted(set(legacy))
        if legacy:
            raise ValueError(
                "Biren generated code must be native SUPA and cannot contain legacy "
                "CUDA token(s): " + ", ".join(legacy)
            )

        missing = []
        if not imports_torch_br:
            missing.append("a torch_br import")
        if not load_inline_calls:
            missing.append("load_inline(...)")
        if "supa_sources" not in load_inline_keywords:
            missing.append("supa_sources=...")
        if "with_supa" not in load_inline_keywords:
            missing.append("with_supa=True")
        if not model_new_found:
            missing.append("class ModelNew")
        if missing:
            raise ValueError(
                "Biren generated code must use the native SUPA extension contract; "
                "missing required token(s): " + ", ".join(missing)
            )

    def compile(self, generated_code, op, dataset_name=None):
        del op, dataset_name
        try:
            self._require_runtime()
            self._validate_native_supa_code(generated_code)
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            self.context["_generated_code"] = generated_code
            return True, None
        except Exception as error:  # pylint: disable=broad-except
            return False, str(error)

    def synchronize(self, device=None):
        self._require_runtime()
        torch.supa.synchronize(device=device or self.device)

    def correctness_execution(self, ref_src):
        del ref_src
        raise NotImplementedError(
            "Standalone op_benchmark does not use backend correctness_execution."
        )

    def time_execution(self, eval_target="ModelNew"):
        from utils.performance import _build_cuda_perf_result, set_perf_seed

        self._require_runtime()
        device = self.device
        get_inputs = self.context["get_inputs"]
        get_init_inputs = self.context["get_init_inputs"]
        model_class = self.context[eval_target]
        set_perf_seed(self.context.get("_op_eval_op_name"))

        warmup = max(0, int(os.environ.get("OP_EVAL_CUDA_NUM_WARMUP", "3")))
        trials = max(1, int(os.environ.get("OP_EVAL_CUDA_NUM_TRIALS", "5")))
        discard_first = max(
            0,
            int(os.environ.get("OP_EVAL_CUDA_DISCARD_FIRST", "1")),
        )

        inputs = get_inputs()
        inputs = [
            item.to(device) if isinstance(item, torch.Tensor) else item
            for item in inputs
        ]
        init_inputs = get_init_inputs()
        init_inputs = [
            item.to(device) if isinstance(item, torch.Tensor) else item
            for item in init_inputs
        ]

        elapsed_times_ms = []
        with torch.no_grad():
            torch.supa.set_device(device.index or 0)
            model = model_class(*init_inputs).to(device).eval()

            def run_once():
                model(*inputs)

            for _ in range(warmup):
                run_once()
                torch.supa.synchronize(device=device)

            torch.supa.empty_cache()
            for trial_index in range(trials + discard_first):
                torch.supa.synchronize(device=device)
                start_event = torch.supa.Event(enable_timing=True)
                end_event = torch.supa.Event(enable_timing=True)
                start_event.record()
                run_once()
                end_event.record()
                torch.supa.synchronize(device=device)
                if trial_index >= discard_first:
                    elapsed_times_ms.append(
                        float(start_event.elapsed_time(end_event))
                    )

        return elapsed_times_ms, _build_cuda_perf_result(elapsed_times_ms)

    def cleanup(self):
        self.context = {}
        if self.device is None or not hasattr(torch, "supa"):
            return
        try:
            torch.supa.set_device(self.device.index or 0)
            torch.supa.empty_cache()
            torch.supa.reset_peak_memory_stats(device=self.device)
            torch.supa.synchronize(device=self.device)
        except Exception:
            pass
