from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.state_space.selective_scan.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference Mamba selective scan over a full sequence."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.delta_softplus = bool(shape["delta_softplus"])
        self.has_z = bool(shape["has_z"])
        self.has_D = bool(shape["has_D"])
        self.use_initial_state = bool(shape["use_initial_state"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        delta_bias: torch.Tensor,
        initial_state: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = D if self.has_D else None
        z = z if self.has_z else None
        initial_state = initial_state if self.use_initial_state else None
        has_initial_state = has_initial_state if self.use_initial_state else None

        batch, dim, seqlen = u.shape
        dstate = A.shape[1]
        n_groups = B.shape[1]
        ratio = dim // n_groups

        u_compute = u.to(self.compute_dtype)
        delta_compute = delta.to(self.compute_dtype)
        delta_compute = delta_compute + delta_bias.to(self.compute_dtype).view(1, dim, 1)
        if self.delta_softplus:
            delta_compute = torch.where(
                delta_compute <= 20.0,
                F.softplus(delta_compute),
                delta_compute,
            )

        state = torch.zeros((batch, dim, dstate), dtype=self.compute_dtype, device=u.device)
        if initial_state is not None:
            state = initial_state.to(self.compute_dtype).clone()
        if has_initial_state is not None:
            mask = has_initial_state.to(torch.bool).view(batch, 1, 1)
            state = torch.where(mask, state, torch.zeros_like(state))

        return self._scan_sequence(u_compute, delta_compute, A, B, C, D, z, state, ratio)

    def _scan_sequence(
        self,
        u_compute: torch.Tensor,
        delta_compute: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        state: torch.Tensor,
        ratio: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, dim, seqlen = u_compute.shape
        out = torch.zeros_like(u_compute)
        A_compute = A.to(self.compute_dtype)
        B_compute = B.to(self.compute_dtype)
        C_compute = C.to(self.compute_dtype)
        D_compute = None if D is None else D.to(self.compute_dtype)
        z_compute = None if z is None else z.to(self.compute_dtype)

        for t in range(seqlen):
            for d in range(dim):
                group = d // ratio
                dt_t = delta_compute[:, d, t].unsqueeze(-1)
                decay = torch.exp(A_compute[d].unsqueeze(0) * dt_t)
                input_term = B_compute[:, group, :, t] * (
                    u_compute[:, d, t].unsqueeze(-1) * dt_t
                )
                state[:, d, :] = state[:, d, :] * decay + input_term
                out[:, d, t] = (state[:, d, :] * C_compute[:, group, :, t]).sum(dim=-1)

            if D_compute is not None:
                out[:, :, t] = out[:, :, t] + u_compute[:, :, t] * D_compute.view(1, dim)
            if z_compute is not None:
                out[:, :, t] = out[:, :, t] * (
                    z_compute[:, :, t] * torch.sigmoid(z_compute[:, :, t])
                )

        return out.to(self.output_dtype), state.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    dim = int(shape["dim"])
    seqlen = int(shape["seqlen"])
    dstate = int(shape["dstate"])
    n_groups = int(shape["n_groups"])
    u = torch.randn((batch_size, dim, seqlen), dtype=torch.float32).to(dtype)
    delta = torch.randn((batch_size, dim, seqlen), dtype=torch.float32).to(dtype)
    A = -torch.rand((dim, dstate), dtype=torch.float32).to(dtype)
    B = torch.randn((batch_size, n_groups, dstate, seqlen), dtype=torch.float32).to(dtype)
    C = torch.randn((batch_size, n_groups, dstate, seqlen), dtype=torch.float32).to(dtype)
    D = torch.randn((dim,), dtype=torch.float32).to(dtype)
    z = torch.randn((batch_size, dim, seqlen), dtype=torch.float32).to(dtype)
    delta_bias = torch.randn((dim,), dtype=torch.float32).to(dtype)
    initial_state = torch.randn((batch_size, dim, dstate), dtype=torch.float32).to(dtype)
    has_initial_state = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
    return u, delta, A, B, C, D, z, delta_bias, initial_state, has_initial_state
