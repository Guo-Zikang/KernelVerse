from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.merge_attn_states.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Merge two attention outputs with numerically stable LSE weights."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> torch.Tensor:
        prefix_compute = prefix_output.to(self.compute_dtype)
        suffix_compute = suffix_output.to(self.compute_dtype)
        prefix_lse_compute = prefix_lse.to(self.compute_dtype)
        suffix_lse_compute = suffix_lse.to(self.compute_dtype)
        max_lse = torch.maximum(prefix_lse_compute, suffix_lse_compute)
        p = torch.exp(prefix_lse_compute - max_lse)
        q = torch.exp(suffix_lse_compute - max_lse)
        denom = p + q
        out = (
            prefix_compute * (p / denom).unsqueeze(-1)
            + suffix_compute * (q / denom).unsqueeze(-1)
        )
        return out.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dims = (
        int(shape["num_tokens"]),
        int(shape["num_heads"]),
        int(shape["head_size"]),
    )
    dtype = _resolve_dtype(shape["dtype"])
    prefix_output = torch.randn(dims, dtype=torch.float32).to(dtype)
    suffix_output = torch.randn(dims, dtype=torch.float32).to(dtype)
    prefix_lse = torch.randn((dims[0], dims[1]), dtype=torch.float32)
    suffix_lse = torch.randn((dims[0], dims[1]), dtype=torch.float32)
    return prefix_output, prefix_lse, suffix_output, suffix_lse
