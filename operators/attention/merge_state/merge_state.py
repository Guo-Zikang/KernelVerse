from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.merge_state.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Merge two attention states and their LSE scores."""

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
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_a_compute = v_a.to(self.compute_dtype)
        v_b_compute = v_b.to(self.compute_dtype)
        s_a_compute = s_a.float()
        s_b_compute = s_b.float()
        max_lse = torch.maximum(s_a_compute, s_b_compute)
        p = torch.exp(s_a_compute - max_lse)
        q = torch.exp(s_b_compute - max_lse)
        denom = p + q
        v_merged = (
            v_a_compute * (p / denom).to(self.compute_dtype).unsqueeze(-1)
            + v_b_compute * (q / denom).to(self.compute_dtype).unsqueeze(-1)
        )
        s_merged = torch.log(denom) + max_lse
        return v_merged.to(self.output_dtype), s_merged


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
    v_a = torch.randn(dims, dtype=torch.float32).to(dtype)
    v_b = torch.randn(dims, dtype=torch.float32).to(dtype)
    s_a = torch.randn((dims[0], dims[1]), dtype=torch.float32)
    s_b = torch.randn((dims[0], dims[1]), dtype=torch.float32)
    return v_a, s_a, v_b, s_b
