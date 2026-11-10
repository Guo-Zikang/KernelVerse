"""Build and load generated Ascend C operators through the CANN toolchain."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


REQUIRED_SOURCE_FIELDS = (
    "project_json_src",
    "host_tiling_src",
    "host_operator_src",
    "kernel_src",
    "python_bind_src",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_root() -> Path:
    configured = os.environ.get("ASCEND_OP_TEMPLATE_ROOT")
    return Path(configured) if configured else _repo_root() / "backends" / "ascend_op_template"


def _pascal_operator_name(op_name: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", op_name) if part]
    if not parts:
        raise ValueError(f"Could not derive an Ascend C operator name from {op_name!r}.")
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Custom"


def _snake_operator_name(op_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", op_name).strip("_").lower()
    if not normalized:
        raise ValueError(f"Could not normalize Ascend C operator name {op_name!r}.")
    return f"{normalized}_custom"


def _declared_operator_name(project_json_src: str) -> str:
    try:
        payload = json.loads(project_json_src)
    except json.JSONDecodeError as exc:
        raise ValueError(f"project_json_src is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError("project_json_src must be a non-empty JSON array of operator objects.")
    declared = payload[0].get("op")
    if not isinstance(declared, str) or not declared:
        raise ValueError("project_json_src[0].op must be a non-empty string.")
    return declared


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    timeout = int(os.environ.get("OP_EVAL_ASCEND_BUILD_TIMEOUT_SECONDS", "600"))
    try:
        subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required Ascend build command was not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Ascend build command timed out after {timeout} seconds: {' '.join(command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part).strip()
        if len(output) > 20000:
            output = output[-20000:]
        raise RuntimeError(
            f"Ascend build command failed ({exc.returncode}): {' '.join(command)}\n{output}"
        ) from exc


@contextmanager
def _build_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prepend_env_path(name: str, value: str) -> None:
    existing = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if value not in existing:
        existing.insert(0, value)
    os.environ[name] = ":".join(existing)


def _stage_custom_opp(project_dir: Path, workspace: Path) -> tuple[Path, Path]:
    installers = sorted((project_dir / "build_out").glob("custom_opp_*.run"))
    if not installers:
        raise RuntimeError(f"No custom_opp_*.run was produced under {project_dir / 'build_out'}.")
    opp_root = workspace / "opp"
    if opp_root.exists():
        shutil.rmtree(opp_root)
    opp_root.mkdir(parents=True)
    installers[0].chmod(installers[0].stat().st_mode | 0o111)
    _run(
        [str(installers[0]), "--quiet", f"--install-path={opp_root}"],
        cwd=project_dir / "build_out",
    )
    customize = opp_root / "vendors" / "customize"
    custom_lib = customize / "op_api" / "lib"
    libcust = custom_lib / "libcust_opapi.so"
    if not libcust.exists():
        raise RuntimeError(f"Ascend custom operator library was not produced: {libcust}")
    libopapi = custom_lib / "libopapi.so"
    if libopapi.exists() or libopapi.is_symlink():
        libopapi.unlink()
    libopapi.symlink_to(libcust.name)
    return customize, custom_lib


def _activate_build(build_info: dict[str, str], context: dict[str, Any]) -> None:
    custom_opp = build_info["custom_opp_path"]
    custom_lib = build_info["custom_lib_path"]
    pybind_path = build_info["pybind_path"]
    module_name = build_info["pybind_module"]

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp)
    _prepend_env_path("LD_LIBRARY_PATH", custom_lib)
    _prepend_env_path("PYTHONPATH", pybind_path)
    if pybind_path not in sys.path:
        sys.path.insert(0, pybind_path)
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    sys.modules["custom_ops_lib"] = module
    context["custom_ops_lib"] = module

    model_src = context.get("model_src")
    if model_src is not None:
        if not isinstance(model_src, str):
            raise TypeError("model_src must be a string when provided.")
        exec(compile(model_src, "<ascend-model-src>", "exec"), context)
    if context.get("ModelNew") is None:
        raise RuntimeError(
            "Ascend C generated code must define ModelNew at top level or inside model_src."
        )


def _cached_build_is_complete(build_info: Any) -> bool:
    if not isinstance(build_info, dict):
        return False
    required_directories = (
        "workspace_root",
        "op_project_dir",
        "custom_opp_path",
        "custom_lib_path",
        "pybind_path",
    )
    for key in required_directories:
        value = build_info.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_dir():
            return False
    module_name = build_info.get("pybind_module")
    if not isinstance(module_name, str) or not module_name:
        return False
    custom_lib = Path(build_info["custom_lib_path"])
    if not (custom_lib / "libcust_opapi.so").is_file():
        return False
    pybind_path = Path(build_info["pybind_path"])
    return any(pybind_path.glob(f"{module_name}*.so"))


def _cache_identity(generated_code: str, target: str) -> str:
    runtime = []
    try:
        import torch
        import torch_npu

        runtime = [torch.__version__, getattr(torch_npu, "__version__", "unknown")]
    except Exception:
        runtime = ["unknown-runtime"]
    material = "\0".join([generated_code, target, *runtime])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _ascend_cache_root() -> Path:
    configured = os.environ.get("TORCH_EXTENSIONS_DIR")
    if configured:
        root = Path(configured).expanduser().resolve()
        # The evaluator isolates CUDA extension caches by device. Ascend C build
        # products are device-index independent, so share one locked cache per op.
        if root.name.startswith("device_") and root.parent.name == "torch_extensions":
            return root.parent / "ascendc_shared"
        return root / "ascendc_shared"
    configured = os.environ.get("ASCEND_OP_RUNS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "op_benchmark" / "ascendc"


def compile_ascendc_generated(
    generated_code: str,
    op_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Compile an Ascend C submission and populate context with ModelNew."""
    compile(generated_code, "<generated-ascendc-operator>", "exec")
    exec(generated_code, context)
    for field in REQUIRED_SOURCE_FIELDS:
        if not isinstance(context.get(field), str) or not context[field].strip():
            raise ValueError(f"Ascend C generated code must define non-empty string {field}.")

    op_pascal = _pascal_operator_name(op_name)
    op_snake = _snake_operator_name(op_name)
    declared_name = _declared_operator_name(context["project_json_src"])
    if declared_name != op_pascal:
        raise ValueError(
            f"project_json_src declares {declared_name!r}; expected {op_pascal!r} for {op_name}."
        )

    target = os.environ.get("ASCENDC_DEVICE", "ai_core-Ascend910B3")
    digest = _cache_identity(generated_code, target)
    cache_root = _ascend_cache_root()
    workspace = cache_root / f"{op_snake}_{digest}"
    marker = workspace / "build_info.json"
    lock_path = workspace.parent / f".{workspace.name}.lock"

    with _build_lock(lock_path):
        if marker.exists():
            try:
                cached_build_info = json.loads(marker.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached_build_info = None
            if _cached_build_is_complete(cached_build_info):
                build_info = cached_build_info
            else:
                marker.unlink(missing_ok=True)

        if not marker.exists():
            template_root = _template_root()
            if not (template_root / "CppExtension").is_dir():
                raise RuntimeError(
                    f"Ascend C template is missing CppExtension assets: {template_root}"
                )
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True)
            shutil.copytree(template_root / "CppExtension", workspace / "CppExtension")

            project_json_path = workspace / f"{op_snake}.json"
            project_json_path.write_text(context["project_json_src"], encoding="utf-8")
            _run(
                [
                    "msopgen",
                    "gen",
                    "-i",
                    project_json_path.name,
                    "-c",
                    target,
                    "-lan",
                    "cpp",
                    "-out",
                    op_pascal,
                ],
                cwd=workspace,
            )

            project_dir = workspace / op_pascal
            if not project_dir.is_dir():
                raise RuntimeError(f"msopgen did not create expected project directory: {project_dir}")
            (project_dir / "op_host" / f"{op_snake}_tiling.h").write_text(
                context["host_tiling_src"], encoding="utf-8"
            )
            (project_dir / "op_host" / f"{op_snake}.cpp").write_text(
                context["host_operator_src"], encoding="utf-8"
            )
            (project_dir / "op_kernel" / f"{op_snake}.cpp").write_text(
                context["kernel_src"], encoding="utf-8"
            )
            (workspace / "CppExtension" / "csrc" / "op.cpp").write_text(
                context["python_bind_src"], encoding="utf-8"
            )

            _run(["bash", "build.sh"], cwd=project_dir)
            custom_opp, custom_lib = _stage_custom_opp(project_dir, workspace)
            module_name = f"custom_ops_lib_{op_snake}_{digest}"
            pybind_path = workspace / "CppExtension" / "pybind_install"
            pybind_env = os.environ.copy()
            pybind_env.update(
                {
                    "ASCEND_CUSTOM_OPP_PATH": str(custom_opp),
                    "CUSTOM_OPS_MODULE_NAME": module_name,
                    "PYBIND_DEST": str(pybind_path),
                    "PYTHON_BIN": sys.executable,
                }
            )
            _prepend_env_path("LD_LIBRARY_PATH", str(custom_lib))
            pybind_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
            _run(
                ["bash", "build_and_run.sh"],
                cwd=workspace / "CppExtension",
                env=pybind_env,
            )
            build_info = {
                "workspace_root": str(workspace),
                "op_project_dir": str(project_dir),
                "op_name": op_snake,
                "op_pascal": op_pascal,
                "ascendc_device": target,
                "custom_opp_path": str(custom_opp),
                "custom_lib_path": str(custom_lib),
                "pybind_path": str(pybind_path),
                "pybind_module": module_name,
            }
            marker.write_text(json.dumps(build_info, indent=2), encoding="utf-8")

    _activate_build(build_info, context)
    context["_op_eval_build_info"] = build_info
    context["_generated_code"] = generated_code
    return build_info
