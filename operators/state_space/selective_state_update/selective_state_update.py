from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.state_space.selective_state_update.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Update selective-scan recurrent state for one decode step."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.dt_softplus = bool(shape["dt_softplus"])
        self.has_z = bool(shape["has_z"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z if self.has_z else None

        dt_eff = dt.to(self.compute_dtype)
        dt_eff = dt_eff + dt_bias.to(self.compute_dtype).unsqueeze(0)
        if self.dt_softplus:
            dt_eff = torch.where(dt_eff <= 20.0, F.softplus(dt_eff), dt_eff)

        state_compute = state.to(self.compute_dtype)
        x_compute = x.to(self.compute_dtype)
        A_compute = A.to(self.compute_dtype)
        B_compute = B.to(self.compute_dtype)
        C_compute = C.to(self.compute_dtype)
        D_compute = D.to(self.compute_dtype)

        dA = torch.exp(A_compute.unsqueeze(0) * dt_eff.unsqueeze(-1))
        dB = B_compute.unsqueeze(2) * dt_eff.unsqueeze(-1)
        new_state = state_compute * dA + dB * x_compute.unsqueeze(-1)

        out = torch.einsum("bhmd,bhd->bhm", new_state, C_compute)
        out = out + x_compute * D_compute.unsqueeze(0)
        if z is not None:
            z_compute = z.to(self.compute_dtype)
            out = out * (z_compute * torch.sigmoid(z_compute))
        return out.to(self.output_dtype), new_state.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = int(shape["num_tokens"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dstate = int(shape["dstate"])

    state = torch.randn((num_tokens, num_heads, head_dim, dstate), dtype=torch.float32).to(dtype)
    x = torch.randn((num_tokens, num_heads, head_dim), dtype=torch.float32).to(dtype)
    dt = torch.randn((num_tokens, num_heads, head_dim), dtype=torch.float32).to(dtype)
    A = -torch.rand((num_heads, head_dim, dstate), dtype=torch.float32).to(dtype)
    B = torch.randn((num_tokens, num_heads, dstate), dtype=torch.float32).to(dtype)
    C = torch.randn((num_tokens, num_heads, dstate), dtype=torch.float32).to(dtype)
    D = torch.randn((num_heads, head_dim), dtype=torch.float32).to(dtype)
    dt_bias = torch.randn((num_heads, head_dim), dtype=torch.float32).to(dtype)
    z = torch.randn((num_tokens, num_heads, head_dim), dtype=torch.float32).to(dtype)
    return state, x, dt, A, B, C, D, dt_bias, z
