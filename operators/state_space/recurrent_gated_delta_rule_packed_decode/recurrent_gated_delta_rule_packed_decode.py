from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.recurrent_gated_delta_rule_packed_decode.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Runs packed single-token recurrent gated delta-rule decode updates."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.key_dim = int(shape["key_dim"])
        self.value_dim = int(shape["value_dim"])
        self.scale = self.key_dim ** (-0.5)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        mixed: torch.Tensor,
        initial_state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k, v = torch.split(
            mixed,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        batch, time, heads, _ = k.shape
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        state = initial_state.to(self.compute_dtype).clone()
        outputs = torch.empty(
            (batch, time, heads, self.value_dim),
            dtype=self.compute_dtype,
            device=k.device,
        )

        for t in range(time):
            kt = k_compute[:, t]
            vt = v_compute[:, t]
            qt = q_compute[:, t]
            decay = torch.full(
                (batch, heads, 1, 1),
                0.5,
                dtype=self.compute_dtype,
                device=k.device,
            )
            state = state * decay + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            outputs[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt * self.scale)

        gathered_state = state.index_select(0, ssm_state_indices.long().clamp(min=0))
        return outputs.to(self.output_dtype), gathered_state.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    num_heads = int(shape["num_heads"])
    key_dim = int(shape["key_dim"])
    value_dim = int(shape["value_dim"])

    mixed = torch.randn(
        (batch_size, 1, num_heads, 2 * key_dim + value_dim),
        dtype=torch.float32,
    ).to(dtype)
    initial_state = torch.randn(
        (batch_size, num_heads, value_dim, key_dim),
        dtype=torch.float32,
    ).to(dtype)
    ssm_state_indices = torch.arange(batch_size, dtype=torch.int32)
    return mixed, initial_state, ssm_state_indices
