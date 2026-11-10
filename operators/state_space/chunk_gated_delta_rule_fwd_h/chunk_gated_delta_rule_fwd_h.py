from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.chunk_gated_delta_rule_fwd_h.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Computes gated delta-rule chunk states and updated value tensors."""

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
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, heads, dim = k.shape
        value_dim = u.shape[-1]
        k_compute = k.to(self.compute_dtype)
        w_compute = w.to(self.compute_dtype)
        u_compute = u.to(self.compute_dtype)
        g_compute = g.to(self.compute_dtype)
        state = initial_state.to(self.compute_dtype).clone()
        h_seq = torch.empty(
            (batch, time, heads, value_dim, dim),
            dtype=self.compute_dtype,
            device=k.device,
        )
        v_new = torch.empty(
            (batch, time, heads, value_dim),
            dtype=self.compute_dtype,
            device=k.device,
        )

        for t in range(time):
            kt = k_compute[:, t]
            ut = u_compute[:, t]
            gt = torch.exp(g_compute[:, t]).unsqueeze(-1).unsqueeze(-1)
            state = state * gt + ut.unsqueeze(-1) * kt.unsqueeze(-2)
            h_seq[:, t] = state
            v_new[:, t] = ut + w_compute[:, t, :, :value_dim]

        return h_seq.to(self.output_dtype), v_new.to(self.output_dtype), state.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    key_dim = int(shape["key_dim"])
    value_dim = int(shape["value_dim"])

    k = torch.randn((batch_size, seq_len, num_heads, key_dim), dtype=torch.float32).to(dtype)
    w = torch.randn((batch_size, seq_len, num_heads, value_dim), dtype=torch.float32).to(dtype)
    u = torch.randn((batch_size, seq_len, num_heads, value_dim), dtype=torch.float32).to(dtype)
    g = (-0.1 * torch.rand(
        (batch_size, seq_len, num_heads),
        dtype=torch.float32,
    )).to(dtype)
    initial_state = torch.randn(
        (batch_size, num_heads, value_dim, key_dim),
        dtype=torch.float32,
    ).to(dtype)
    return k, w, u, g, initial_state
