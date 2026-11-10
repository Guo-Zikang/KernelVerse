"""MetaX MACA backend using the vendor CUDA-compatible PyTorch runtime."""

from __future__ import annotations

import os

import torch

from backends.backend_registry import register_backend
from backends.cuda_backend import CudaBackend


@register_backend("maca")
class MacaBackend(CudaBackend):
    """Compile and evaluate MACA kernels through CUDA-compatible torch APIs."""

    language = "maca"

    def __init__(self):
        # register_backend instantiates the class before its global name is bound,
        # so use an explicit base-class call instead of bare super().
        CudaBackend.__init__(self)
        self._maca_device_applied = False

    def set_device(self, device_id: int):
        self._maca_device_applied = True
        CudaBackend.set_device(self, device_id)

    def get_device(self):
        if not self._maca_device_applied:
            configured_device = os.environ.get("MACA_DEVICE_ID")
            if configured_device is not None:
                self.set_device(int(configured_device))
            else:
                self._maca_device_applied = True
        return self.device

    def get_hardware_name(self):
        configured_name = os.environ.get("MACA_DEVICE_NAME")
        if configured_name:
            return configured_name
        return CudaBackend.get_hardware_name(self)

    def _configure_cuda_arch_list(self):
        """Use the MACA target arch and never inject NVIDIA's static sm_80."""
        if os.environ.get("TORCH_CUDA_ARCH_LIST"):
            return

        configured_arch = os.environ.get("MACA_TORCH_CUDA_ARCH_LIST")
        if configured_arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = configured_arch
            return

        try:
            major, minor = torch.cuda.get_device_capability(self.device)
        except Exception:
            # Vendor cpp_extension builds can infer their native target when the
            # compatibility runtime does not expose CUDA compute capability.
            return
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

