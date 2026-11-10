from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.gemm.addmm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Matrix multiply with addend using torch.addmm."""

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
        mat1: torch.Tensor,
        mat2: torch.Tensor,
    ) -> torch.Tensor:
        input_compute = input_tensor.to(self.compute_dtype)
        mat1_compute = mat1.to(self.compute_dtype)
        mat2_compute = mat2.to(self.compute_dtype)
        output = torch.addmm(
            input_compute,
            mat1_compute,
            mat2_compute,
            beta=self.beta,
            alpha=self.alpha,
        )
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    m_value = int(shape["M"])
    n_value = int(shape["N"])
    k_value = int(shape["K"])
    dtype = _resolve_dtype(shape["dtype"])
    input_tensor = torch.randn((m_value, n_value), dtype=torch.float32).to(dtype)
    mat1 = torch.randn((m_value, k_value), dtype=torch.float32).to(dtype)
    mat2 = torch.randn((k_value, n_value), dtype=torch.float32).to(dtype)
    return input_tensor, mat1, mat2

