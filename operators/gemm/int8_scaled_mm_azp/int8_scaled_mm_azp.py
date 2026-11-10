from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.gemm.int8_scaled_mm_azp.helpers import (
    _resolve_output_dtype,
    validate_shape,
)


def _expand_rows(scale: torch.Tensor, rows: int, dtype: torch.dtype) -> torch.Tensor:
    scale = scale.to(dtype)
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(rows, 1)
    if scale.ndim == 1 and scale.shape[0] == rows:
        return scale.reshape(rows, 1)
    if scale.ndim == 2 and scale.shape == (rows, 1):
        return scale
    return scale.reshape(rows, -1)


def _expand_cols(scale: torch.Tensor, cols: int, dtype: torch.dtype) -> torch.Tensor:
    scale = scale.to(dtype)
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(1, cols)
    if scale.ndim == 1 and scale.shape[0] == cols:
        return scale.reshape(1, cols)
    if scale.ndim == 2 and scale.shape == (1, cols):
        return scale
    return scale.reshape(-1, cols)


def _broadcast_like(value: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    value = value.to(out.dtype)
    if value.numel() == 1:
        return value.reshape(1, 1).expand_as(out)
    if value.ndim == 1 and value.shape[0] == out.shape[-1]:
        return value.reshape(1, -1).expand_as(out)
    if value.shape == out.shape:
        return value
    return value.reshape_as(out)


class Model(nn.Module):
    """Runs scaled INT8 matrix multiplication with optional activation zero-point correction."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.has_azp = bool(shape["has_azp"])
        self.has_bias = bool(shape["has_bias"])
        self.output_dtype = _resolve_output_dtype(shape["output_dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        azp_adj: torch.Tensor,
        azp: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        rows = a.shape[0]
        cols = b.shape[1]

        a_deq = a.to(self.compute_dtype) * _expand_rows(a_scales, rows, self.compute_dtype)
        if self.has_azp:
            b_centered = b.to(self.compute_dtype) - _expand_cols(azp, cols, self.compute_dtype)
        else:
            b_centered = b.to(self.compute_dtype)
        b_deq = b_centered * _expand_cols(b_scales, cols, self.compute_dtype)

        out = a_deq @ b_deq
        out = out + _broadcast_like(azp_adj, out)
        if self.has_bias:
            out = out + _broadcast_like(bias, out)
        return out.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    validate_shape(shape)
    rows = int(shape["M"])
    cols = int(shape["N"])
    inner = int(shape["K"])

    a = torch.randint(-64, 64, (rows, inner), dtype=torch.int8)
    b = torch.randint(-64, 64, (inner, cols), dtype=torch.int8)
    a_scales = torch.rand((rows,), dtype=torch.float32) * 0.02 + 0.001
    b_scales = torch.rand((cols,), dtype=torch.float32) * 0.02 + 0.001
    azp_adj = torch.randn((cols,), dtype=torch.float32) * 0.01
    azp = torch.randint(-8, 8, (cols,), dtype=torch.int32).to(torch.float32)
    bias = torch.randn((cols,), dtype=torch.float32) * 0.01
    return a, b, a_scales, b_scales, azp_adj, azp, bias
