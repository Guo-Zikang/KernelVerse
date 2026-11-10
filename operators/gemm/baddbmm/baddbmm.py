from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.gemm.baddbmm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Batched matrix multiply with addend using torch.baddbmm."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.alpha = float(shape.get("alpha", 1.0))
        self.beta = float(shape.get("beta", 1.0))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
    ) -> torch.Tensor:
        input_compute = input_tensor.to(self.compute_dtype)
        batch1_compute = batch1.to(self.compute_dtype)
        batch2_compute = batch2.to(self.compute_dtype)
        output = torch.baddbmm(
            input_compute,
            batch1_compute,
            batch2_compute,
            beta=self.beta,
            alpha=self.alpha,
        )
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    m_value = int(shape["M"])
    n_value = int(shape["N"])
    k_value = int(shape["K"])
    dtype = _resolve_dtype(shape["dtype"])
    input_tensor = torch.randn(
        (batch_size, m_value, n_value),
        dtype=torch.float32,
    ).to(dtype)
    batch1 = torch.randn(
        (batch_size, m_value, k_value),
        dtype=torch.float32,
    ).to(dtype)
    batch2 = torch.randn(
        (batch_size, k_value, n_value),
        dtype=torch.float32,
    ).to(dtype)
    return input_tensor, batch1, batch2
