from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.kda_output_combine.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Combines KDA attention output with gated value and beta terms."""

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
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        q_compute = q_proj_states.to(self.compute_dtype)
        k_compute = k_proj_states.to(self.compute_dtype)
        v_compute = v_proj_states.to(self.compute_dtype)
        g_compute = g1.to(self.compute_dtype)
        beta_compute = beta.to(self.compute_dtype)
        core_compute = core_attn_out.to(self.compute_dtype)

        gate = torch.sigmoid(g_compute)
        gated_value = gate * v_compute
        beta_mix = beta_compute.unsqueeze(-1) * (q_compute + k_compute) * 0.5
        return (core_compute + gated_value + beta_mix).to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = int(shape["num_tokens"])
    hidden_dim = int(shape["hidden_dim"])

    q = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    k = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    v = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    g1 = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    beta = torch.rand((num_tokens,), dtype=torch.float32).to(dtype)
    out = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    return q, k, v, g1, beta, out
