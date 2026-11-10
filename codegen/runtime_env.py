"""Runtime environment collection for CUDA code generation prompts."""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict

from runner.config import ARCH_LIST
from utils.biren_environment import prepare_biren_environment


CUDA_COMPATIBLE_LANGUAGES = {"cuda", "maca", "tiangai"}


def collect_runtime_env(*, language: str, device_id: int) -> Dict[str, Any]:
    """Collect target runtime details without requiring an available accelerator."""
    normalized_language = str(language).lower()
    configured_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if normalized_language == "maca":
        configured_arch = configured_arch or os.environ.get("MACA_TORCH_CUDA_ARCH_LIST")
    elif normalized_language == "tiangai":
        configured_arch = None
    elif normalized_language == "cuda" and configured_arch is None:
        configured_arch = ";".join(ARCH_LIST)

    env: Dict[str, Any] = {
        "backend": normalized_language,
        "device_id": device_id,
        "gpu_name": None,
        "compute_capability": None,
        "sm": None,
        "torch_version": None,
        "cuda_runtime_version": None,
        "torch_cuda_arch_list": configured_arch,
        "environment_available": False,
    }
    if normalized_language == "maca":
        env.update(
            {
                "maca_home": os.environ.get("MACA_HOME") or os.environ.get("MACA_PATH"),
                "maca_torch_cuda_arch_list": os.environ.get("MACA_TORCH_CUDA_ARCH_LIST"),
            }
        )

    if normalized_language == "tiangai":
        corex_home = (
            os.environ.get("TIANGAI_CUDA_HOME")
            or os.environ.get("COREX_HOME")
            or os.environ.get("COREX_PATH")
            or "/usr/local/corex"
        )
        env.update(
            {
                "corex_home": corex_home,
                "corex_arch": os.environ.get("TIANGAI_CUDA_ARCH", "ivcore11"),
                "compiler": os.environ.get("PYTORCH_NVCC")
                or os.path.join(corex_home, "bin", "clang++"),
            }
        )

    if normalized_language == "biren":
        env.update(
            {
                "supa_runtime_version": None,
                "torch_br_version": None,
                "biren_sdk_root": os.environ.get("BIRENSUPA_ROOT"),
                "compiler": "brcc",
                "compiler_path": None,
            }
        )
        try:
            sdk_root = prepare_biren_environment()
            env["biren_sdk_root"] = sdk_root or env["biren_sdk_root"]
            env["compiler_path"] = shutil.which("brcc")
            import torch
            import torch_br

            env["torch_version"] = torch.__version__
            env["supa_runtime_version"] = getattr(torch_br, "__version__", None)
            env["torch_br_version"] = getattr(torch_br, "__version__", None)
            if not hasattr(torch, "supa") or not torch.supa.is_available():
                return env
            device_count = torch.supa.device_count()
            if device_id >= device_count:
                env["error_message"] = (
                    f"SUPA device_id {device_id} is unavailable; device_count={device_count}."
                )
                return env
            env["gpu_name"] = torch.supa.get_device_name(device=device_id)
            env["environment_available"] = True
        except Exception as error:  # pylint: disable=broad-except
            env["error_message"] = str(error)
        return env

    if normalized_language == "ascendc":
        env.update(
            {
                "accelerator": "Huawei Ascend NPU",
                "ascendc_device": os.environ.get("ASCENDC_DEVICE", "ai_core-Ascend910B3"),
                "ascend_toolkit_home": os.environ.get("ASCEND_TOOLKIT_HOME")
                or "/usr/local/Ascend/ascend-toolkit/latest",
                "cann_version": os.environ.get("ASCEND_TOOLKIT_VERSION"),
                "torch_npu_version": None,
                "compiler": "Ascend C via msopgen",
                "compiler_path": shutil.which("msopgen"),
            }
        )
        try:
            import torch
            import torch_npu

            env["torch_version"] = torch.__version__
            env["torch_npu_version"] = getattr(torch_npu, "__version__", None)
            if not torch_npu.npu.is_available():
                return env
            device_count = torch_npu.npu.device_count()
            if device_id >= device_count:
                env["error_message"] = (
                    f"NPU device_id {device_id} is unavailable; device_count={device_count}."
                )
                return env
            device = torch.device(f"npu:{device_id}")
            env["gpu_name"] = torch_npu.npu.get_device_name(device)
            env["environment_available"] = True
        except Exception as error:  # pylint: disable=broad-except
            env["error_message"] = str(error)
        return env

    if normalized_language not in CUDA_COMPATIBLE_LANGUAGES:
        return env

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_runtime_version"] = torch.version.cuda
        if not torch.cuda.is_available():
            return env

        device_count = torch.cuda.device_count()
        if device_id >= device_count:
            env["error_message"] = (
                f"CUDA device_id {device_id} is unavailable; device_count={device_count}."
            )
            return env

        env["gpu_name"] = torch.cuda.get_device_name(device_id)
        env["environment_available"] = True
        if normalized_language == "tiangai":
            return env
        try:
            capability = torch.cuda.get_device_capability(device_id)
            detected_arch = f"{capability[0]}.{capability[1]}"
            if env["torch_cuda_arch_list"] is None:
                env["torch_cuda_arch_list"] = detected_arch
            env["compute_capability"] = list(capability)
            env["sm"] = f"sm_{capability[0]}{capability[1]}"
        except Exception as capability_error:  # pylint: disable=broad-except
            env["architecture_error_message"] = str(capability_error)
    except Exception as error:  # pylint: disable=broad-except
        env["error_message"] = str(error)

    return env
