from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.moe.apply_shuffle_mul_sum.helpers import (
    _resolve_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """
    Shuffles rows, optionally applies factors, and sums over top-k groups.
    Used to merge routed expert outputs in MoE inference.
    """

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.topk = int(shape["topk"])
        self.has_factors = bool(shape["has_factors"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        permutation: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        m = permutation.numel() // self.topk
        gathered = x.index_select(0, permutation.long()).view(m, self.topk, x.size(1))
        gathered = gathered.to(self.compute_dtype)
        if self.has_factors:
            gathered = gathered * factors.to(self.compute_dtype).view(m, self.topk, 1)
        return gathered.sum(dim=1).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    index_dtype = _resolve_index_dtype(shape["index_dtype"])
    rows = int(shape["m"]) * int(shape["topk"])
    hidden_size = int(shape["hidden_size"])

    x = torch.randn((rows, hidden_size), dtype=torch.float32).to(dtype)
    permutation = torch.randperm(rows, dtype=index_dtype)
    if shape["has_factors"]:
        factors = torch.rand((rows,), dtype=torch.float32).to(dtype)
    else:
        factors = torch.ones((rows,), dtype=dtype)
    return x, permutation, factors
