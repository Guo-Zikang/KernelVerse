"""Huawei Ascend C backend using CANN custom-operator projects."""

from __future__ import annotations

import os
from typing import Any

from backends.backend_registry import register_backend
from utils.ascend_compile_pipeline import compile_ascendc_generated


@register_backend("ascendc")
class AscendCBackend:
    """Compile Ascend C sources and evaluate them through torch_npu."""

    language = "ascendc"

    def __init__(self):
        self.context: dict[str, Any] = {}
        self.device = None

    @staticmethod
    def _runtime():
        import torch
        import torch_npu

        return torch, torch_npu

    def setup_device(self, device_id: int):
        torch, torch_npu = self._runtime()
        torch_npu.npu.set_device(device_id)
        self.device = torch.device(f"npu:{device_id}")
        os.environ["ASCEND_DEVICE_ID"] = str(device_id)
        return self.device

    def set_device(self, device_id: int):
        return self.setup_device(device_id)

    def get_device(self):
        if self.device is None:
            return self.setup_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))
        return self.device

    def get_hardware_name(self):
        _, torch_npu = self._runtime()
        return torch_npu.npu.get_device_name(self.get_device())

    def compile(self, generated_code, op, dataset_name=None):
        del dataset_name
        self.context = {}
        try:
            compile_ascendc_generated(generated_code, op, self.context)
            return True, None
        except Exception as error:  # pylint: disable=broad-except
            return False, str(error)

    def synchronize(self, device=None):
        _, torch_npu = self._runtime()
        torch_npu.npu.synchronize(device=device or self.get_device())

    def time_execution(self, eval_target="ModelNew"):
        from runner.config import NUM_PERF_TRIALS, NUM_WARMUP
        from utils.performance import _build_cuda_perf_result, set_perf_seed

        torch, torch_npu = self._runtime()
        device = self.get_device()
        model_class = self.context[eval_target]
        get_inputs = self.context["get_inputs"]
        get_init_inputs = self.context["get_init_inputs"]
        set_perf_seed(self.context.get("_op_eval_op_name"))

        inputs = [
            item.to(device) if isinstance(item, torch.Tensor) else item
            for item in get_inputs()
        ]
        init_inputs = [
            item.to(device) if isinstance(item, torch.Tensor) else item
            for item in get_init_inputs()
        ]
        warmup = max(0, int(os.environ.get("OP_EVAL_NPU_NUM_WARMUP", str(NUM_WARMUP))))
        trials = max(
            1,
            int(os.environ.get("OP_EVAL_CUDA_NUM_TRIALS", str(NUM_PERF_TRIALS))),
        )
        discard_first = max(0, int(os.environ.get("OP_EVAL_NPU_DISCARD_FIRST", "1")))
        elapsed_times_ms = []
        with torch.no_grad():
            model = model_class(*init_inputs).to(device).eval()
            for _ in range(warmup):
                model(*inputs)
                torch_npu.npu.synchronize(device=device)
            for trial_index in range(trials + discard_first):
                torch_npu.npu.synchronize(device=device)
                start = torch_npu.npu.Event(enable_timing=True)
                end = torch_npu.npu.Event(enable_timing=True)
                start.record()
                model(*inputs)
                end.record()
                torch_npu.npu.synchronize(device=device)
                if trial_index >= discard_first:
                    elapsed_times_ms.append(float(start.elapsed_time(end)))

        # Keep the same result builder and JSON schema used by CUDA-compatible backends.
        return elapsed_times_ms, _build_cuda_perf_result(elapsed_times_ms)

    def cleanup(self):
        self.context = {}
        if self.device is None:
            return
        _, torch_npu = self._runtime()
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize(device=self.device)
