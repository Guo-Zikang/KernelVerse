"""Iluvatar Tiangai/CoreX backend for generated CUDA operator code."""

from __future__ import annotations

import os

from backends.backend_registry import register_backend
from backends.cuda_backend import CudaBackend


@register_backend("tiangai")
class TiangaiBackend(CudaBackend):
    """Compile CUDA-compatible kernels with the CoreX clang++ toolchain."""

    language = "tiangai"

    DEFAULT_COREX_HOME = "/usr/local/corex"
    DEFAULT_ARCH = "ivcore11"
    DEFAULT_GCC_VERSION = "9"

    def __init__(self):
        # register_backend instantiates the class before its global name is bound.
        CudaBackend.__init__(self)

    def _configure_cuda_arch_list(self):
        """CoreX uses ivcore architecture flags, not NVIDIA SM values."""
        return None

    def _configure_toolchain_environment(self) -> str:
        corex_home = (
            os.environ.get("TIANGAI_CUDA_HOME")
            or os.environ.get("COREX_HOME")
            or os.environ.get("COREX_PATH")
            or self.DEFAULT_COREX_HOME
        )
        if corex_home and os.path.isdir(corex_home):
            os.environ.setdefault("CUDA_HOME", corex_home)
            os.environ.setdefault("CUDA_PATH", corex_home)

        if not os.environ.get("PYTORCH_NVCC"):
            compiler = os.path.join(corex_home, "bin", "clang++")
            if os.path.exists(compiler):
                os.environ["PYTORCH_NVCC"] = compiler

        configured_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if configured_arch and "ivcore" not in configured_arch.lower():
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)

        gcc_version = os.environ.get("TIANGAI_GCC_VER", self.DEFAULT_GCC_VERSION)
        include_candidates = [
            f"/usr/include/c++/{gcc_version}",
            f"/usr/include/x86_64-linux-gnu/c++/{gcc_version}",
            f"/usr/lib/x86_64-linux-gnu/c++/{gcc_version}",
        ]
        library_candidates = [
            f"/usr/lib/gcc/x86_64-linux-gnu/{gcc_version}",
            "/usr/lib/x86_64-linux-gnu",
            os.path.join(corex_home, "lib64"),
        ]

        include_entries = [
            entry for entry in os.environ.get("CPLUS_INCLUDE_PATH", "").split(":") if entry
        ]
        for path in include_candidates:
            if os.path.isdir(path) and path not in include_entries:
                include_entries.append(path)
        if include_entries:
            os.environ["CPLUS_INCLUDE_PATH"] = ":".join(include_entries)

        library_entries = [
            entry for entry in os.environ.get("LIBRARY_PATH", "").split(":") if entry
        ]
        for path in library_candidates:
            if os.path.isdir(path) and path not in library_entries:
                library_entries.append(path)
        if library_entries:
            os.environ["LIBRARY_PATH"] = ":".join(library_entries)

        ld_entries = [
            entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry
        ]
        corex_library = os.path.join(corex_home, "lib64")
        if os.path.isdir(corex_library) and corex_library not in ld_entries:
            ld_entries.insert(0, corex_library)
        if ld_entries:
            os.environ["LD_LIBRARY_PATH"] = ":".join(ld_entries)
        return corex_home

    @classmethod
    def _rewrite_cuda_to_tiangai(cls, code: str) -> str:
        """Patch cpp_extension calls with CoreX-compatible compile/link flags."""
        marker = "_op_benchmark_tiangai_patch_installed"
        prelude = f'''
import os as _op_benchmark_os
import torch.utils.cpp_extension as _op_benchmark_ce
if not getattr(_op_benchmark_ce, "{marker}", False):
    _op_benchmark_original_load_inline = _op_benchmark_ce.load_inline
    _op_benchmark_original_load = _op_benchmark_ce.load
    _op_benchmark_corex_home = _op_benchmark_os.environ.get("TIANGAI_CUDA_HOME") or _op_benchmark_os.environ.get("COREX_HOME") or _op_benchmark_os.environ.get("COREX_PATH") or "{cls.DEFAULT_COREX_HOME}"
    _op_benchmark_arch = _op_benchmark_os.environ.get("TIANGAI_CUDA_ARCH", "{cls.DEFAULT_ARCH}")
    _op_benchmark_gcc_version = _op_benchmark_os.environ.get("TIANGAI_GCC_VER", "{cls.DEFAULT_GCC_VERSION}")
    _op_benchmark_include_dirs = [
        f"/usr/include/c++/{{_op_benchmark_gcc_version}}",
        f"/usr/include/x86_64-linux-gnu/c++/{{_op_benchmark_gcc_version}}",
        f"/usr/lib/x86_64-linux-gnu/c++/{{_op_benchmark_gcc_version}}",
    ]
    _op_benchmark_library_dirs = [
        f"/usr/lib/gcc/x86_64-linux-gnu/{{_op_benchmark_gcc_version}}",
        "/usr/lib/x86_64-linux-gnu",
    ]

    def _op_benchmark_normalize_cuda_flags(flags):
        flags = list(flags or [])
        use_fast_math = False
        normalized = []
        nvcc_only_flags = (
            "--extra-device-vectorization", "-maxrregcount", "--maxrregcount",
            "-lineinfo", "--generate-line-info", "--ptxas-options", "-Xptxas",
            "-gencode", "-arch=", "--gpu-architecture", "-rdc=",
            "--relocatable-device-code", "--default-stream",
        )
        for flag in flags:
            if flag == "--use_fast_math":
                use_fast_math = True
                continue
            if any(str(flag) == prefix or str(flag).startswith(prefix) for prefix in nvcc_only_flags):
                continue
            normalized.append(flag)

        drop_fast_math = _op_benchmark_os.environ.get("TIANGAI_DROP_FAST_MATH", "0") == "1"
        if use_fast_math and not drop_fast_math and "-ffast-math" not in normalized:
            normalized.append("-ffast-math")
        if "-x" not in normalized:
            normalized.extend(["-x", "ivcore"])
        elif normalized.index("-x") + 1 >= len(normalized):
            normalized.append("ivcore")
        if "-Wno-unknown-cuda-version" not in normalized:
            normalized.append("-Wno-unknown-cuda-version")
        if not any(str(flag).startswith("--cuda-gpu-arch=") for flag in normalized):
            normalized.append(f"--cuda-gpu-arch={{_op_benchmark_arch}}")
        if _op_benchmark_corex_home:
            if not any(str(flag).startswith("--cuda-path=") for flag in normalized):
                normalized.append(f"--cuda-path={{_op_benchmark_corex_home}}")
            include_flag = f"-I{{_op_benchmark_corex_home}}/include"
            if include_flag not in normalized:
                normalized.append(include_flag)
        for include_dir in _op_benchmark_include_dirs:
            if include_dir and _op_benchmark_os.path.isdir(include_dir):
                include_flag = f"-I{{include_dir}}"
                if include_flag not in normalized:
                    normalized.append(include_flag)
        return normalized

    def _op_benchmark_normalize_link_flags(flags):
        normalized = list(flags or [])
        if _op_benchmark_corex_home:
            library_dir = f"{{_op_benchmark_corex_home}}/lib64"
            library_flag = f"-L{{library_dir}}"
            if library_flag not in normalized:
                normalized.append(library_flag)
            if "-lcudart" not in normalized:
                normalized.append("-lcudart")
            rpath_flag = f"-Wl,-rpath,{{library_dir}}"
            if rpath_flag not in normalized:
                normalized.append(rpath_flag)
        for library_dir in _op_benchmark_library_dirs:
            if library_dir and _op_benchmark_os.path.isdir(library_dir):
                library_flag = f"-L{{library_dir}}"
                if library_flag not in normalized:
                    normalized.append(library_flag)
        return normalized

    def _op_benchmark_load_inline(*args, **kwargs):
        kwargs["extra_cuda_cflags"] = _op_benchmark_normalize_cuda_flags(
            kwargs.get("extra_cuda_cflags")
        )
        kwargs["extra_ldflags"] = _op_benchmark_normalize_link_flags(
            kwargs.get("extra_ldflags")
        )
        if kwargs.get("with_cuda") is None:
            kwargs["with_cuda"] = True
        return _op_benchmark_original_load_inline(*args, **kwargs)

    def _op_benchmark_load(*args, **kwargs):
        kwargs["extra_cuda_cflags"] = _op_benchmark_normalize_cuda_flags(
            kwargs.get("extra_cuda_cflags")
        )
        kwargs["extra_ldflags"] = _op_benchmark_normalize_link_flags(
            kwargs.get("extra_ldflags")
        )
        if kwargs.get("with_cuda") is None:
            kwargs["with_cuda"] = True
        return _op_benchmark_original_load(*args, **kwargs)

    _op_benchmark_ce.load_inline = _op_benchmark_load_inline
    _op_benchmark_ce.load = _op_benchmark_load
    _op_benchmark_ce.{marker} = True
'''
        return prelude + "\n" + code

    def compile(self, generated_code, op, dataset_name=None):
        del op, dataset_name
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        try:
            compile(generated_code, "<string>", "exec")
            self._configure_toolchain_environment()
            rewritten_code = self._rewrite_cuda_to_tiangai(generated_code)
            compile(rewritten_code, "<string>", "exec")
            exec(rewritten_code, self.context)
            self.context["_generated_code"] = rewritten_code
            return True, None
        except Exception as error:  # pylint: disable=broad-except
            return False, str(error)
