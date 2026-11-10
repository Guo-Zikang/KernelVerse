from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.chunk_fwd_o.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Combines chunk causal attention with gated recurrent state output."""

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
        h: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        h_compute = h.to(self.compute_dtype)
        g_compute = g.to(self.compute_dtype)

        batch, time, heads, _ = q_compute.shape
        scores = torch.einsum("bthd,bshd->bhts", q_compute, k_compute) * self.scale
        mask = torch.tril(torch.ones((time, time), dtype=torch.bool, device=q.device))
        scores = torch.where(
            mask.view(1, 1, time, time),
            scores,
            torch.full((), float("-inf"), dtype=scores.dtype, device=scores.device),
        )
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum("bhts,bshv->bthv", probs, v_compute)

        state_out = torch.einsum("bthvk,bthk->bthv", h_compute, q_compute)
        gate = torch.sigmoid(g_compute).unsqueeze(-1)
        out = attn_out + gate * state_out
        return out.reshape(batch, time, heads, -1).to(self.output_dtype)


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

    q = torch.randn((batch_size, seq_len, num_heads, key_dim), dtype=torch.float32).to(dtype)
    k = torch.randn((batch_size, seq_len, num_heads, key_dim), dtype=torch.float32).to(dtype)
    v = torch.randn((batch_size, seq_len, num_heads, value_dim), dtype=torch.float32).to(dtype)
    h = torch.randn(
        (batch_size, seq_len, num_heads, value_dim, key_dim),
        dtype=torch.float32,
    ).to(dtype)
    g = torch.randn((batch_size, seq_len, num_heads), dtype=torch.float32).to(dtype)
    return q, k, v, h, g
