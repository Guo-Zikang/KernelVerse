"""CUDA backend for standalone op_benchmark."""

import os

import torch

from backends.backend_registry import Backend, register_backend
from runner.config import ARCH_LIST
from utils.performance import time_execution_event_template


@register_backend("cuda")
class CudaBackend(Backend):
    """Execute and time operators on CUDA devices."""

    def __init__(self):
        self.context = {}
        self.device = torch.device("cuda:0")

    def get_device(self):
        return self.device

    def set_device(self, device_id: int):
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)

    def get_hardware_name(self):
        return torch.cuda.get_device_name(device=self.device)

    def _configure_cuda_arch_list(self):
        """Set the default NVIDIA architecture when the user did not pin one."""
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(ARCH_LIST)

    def compile(self, generated_code, op, dataset_name=None):
        del op, dataset_name
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        self._configure_cuda_arch_list()
        try:
            compile(generated_code, "<string>", "exec")
            exec(generated_code, self.context)
            self.context["_generated_code"] = generated_code
            return True, None
        except Exception as error:  # pylint: disable=broad-except
            return False, str(error)

    def synchronize(self, device=None):
        torch.cuda.synchronize(device=device or self.device)

    def correctness_execution(self, ref_src):
        del ref_src
        raise NotImplementedError("Standalone op_benchmark does not use backend correctness_execution.")

    def time_execution(self, eval_target="ModelNew"):
        synchronize = torch.cuda.synchronize
        event_class = torch.cuda.Event
        return time_execution_event_template(
            self.context,
            self.device,
            synchronize,
            event_class,
            eval_target,
        )

    def cleanup(self):
        self.context = {}
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=self.device)
            torch.cuda.synchronize(device=self.device)
