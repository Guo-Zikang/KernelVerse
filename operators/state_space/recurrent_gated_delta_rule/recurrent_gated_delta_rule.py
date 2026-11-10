from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.recurrent_gated_delta_rule.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Runs recurrent gated delta-rule state updates over a token sequence."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.scale = int(shape["key_dim"]) ** (-0.5)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gate_logits: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, heads, _ = k.shape
        value_dim = v.shape[-1]
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        beta_compute = beta.to(self.compute_dtype)
        gate_compute = gate_logits.to(self.compute_dtype)
        state = initial_state.to(self.compute_dtype).clone()
        outputs = torch.empty(
            (batch, time, heads, value_dim),
            dtype=self.compute_dtype,
            device=k.device,
        )

        for t in range(time):
            kt = k_compute[:, t]
            vt = v_compute[:, t]
            qt = q_compute[:, t]
            decay = torch.sigmoid(gate_compute[:, t]).unsqueeze(-1).unsqueeze(-1)
            update = (beta_compute[:, t].unsqueeze(-1) * vt).unsqueeze(-1) * kt.unsqueeze(-2)
            state = state * decay + update
            outputs[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt * self.scale)

        return outputs.to(self.output_dtype), state.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    key_dim = int(shape["key_dim"])
    value_dim = int(shape["value_dim"])

    q = torch.randn((batch_size, seq_len, num_heads, key_dim), dtype=torch.float32).to(dtype)
    k = torch.randn((batch_size, seq_len, num_heads, key_dim), dtype=torch.float32).to(dtype)
    v = torch.randn((batch_size, seq_len, num_heads, value_dim), dtype=torch.float32).to(dtype)
    beta = torch.sigmoid(torch.randn((batch_size, seq_len, num_heads), dtype=torch.float32)).to(dtype)
    gate_logits = torch.randn((batch_size, seq_len, num_heads), dtype=torch.float32).to(dtype)
    initial_state = torch.randn(
        (batch_size, num_heads, value_dim, key_dim),
        dtype=torch.float32,
    ).to(dtype)
    return q, k, v, beta, gate_logits, initial_state
