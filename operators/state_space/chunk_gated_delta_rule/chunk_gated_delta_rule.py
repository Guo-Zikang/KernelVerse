from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.chunk_gated_delta_rule.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Runs chunked gated delta-rule attention and returns the final state."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
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
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, heads, dim = k.shape
        value_dim = v.shape[-1]
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        g_compute = g.to(self.compute_dtype)
        beta_compute = beta.to(self.compute_dtype)

        g_cumsum = torch.empty_like(g_compute)
        for start in range(0, time, self.chunk_size):
            end = min(time, start + self.chunk_size)
            g_cumsum[:, start:end] = torch.cumsum(g_compute[:, start:end], dim=1)

        A = torch.zeros(
            (batch, time, heads, self.chunk_size),
            dtype=self.compute_dtype,
            device=k.device,
        )
        for start in range(0, time, self.chunk_size):
            end = min(time, start + self.chunk_size)
            cur = k_compute[:, start:end] * torch.exp(g_cumsum[:, start:end]).unsqueeze(-1)
            cur_beta = beta_compute[:, start:end].unsqueeze(-1)
            gram = torch.einsum("bthd,bshd->bhts", cur * cur_beta, cur)
            A[:, start:end, :, : end - start] = gram.permute(0, 2, 1, 3)

        weight = beta_compute.unsqueeze(-1) * torch.exp(g_cumsum).unsqueeze(-1)
        u = v_compute * weight
        k_padded = torch.zeros(
            k_compute.shape[:-1] + (self.chunk_size,),
            dtype=self.compute_dtype,
            device=k.device,
        )
        copy_width = min(self.chunk_size, dim)
        k_padded[..., :copy_width] = k_compute[..., :copy_width]
        w = A * k_padded

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
            ut = u[:, t]
            gt = torch.exp(g_cumsum[:, t]).unsqueeze(-1).unsqueeze(-1)
            state = state * gt + ut.unsqueeze(-1) * kt.unsqueeze(-2)
            h_seq[:, t] = state
            v_new[:, t] = ut + w[:, t, :, :value_dim]

        scores = torch.einsum("bthd,bshd->bhts", q_compute, k_compute) * self.scale
        mask = torch.tril(torch.ones((time, time), dtype=torch.bool, device=q.device))
        scores = torch.where(
            mask.view(1, 1, time, time),
            scores,
            torch.full((), float("-inf"), dtype=scores.dtype, device=scores.device),
        )
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum("bhts,bshv->bthv", probs, v_new)
        state_out = torch.einsum("bthvk,bthk->bthv", h_seq, q_compute)
        gate = torch.sigmoid(g_cumsum).unsqueeze(-1)
        out = attn_out + gate * state_out
        return out.to(self.output_dtype), state.to(self.output_dtype)


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
    g = torch.randn((batch_size, seq_len, num_heads), dtype=torch.float32).to(dtype)
    beta = torch.sigmoid(torch.randn((batch_size, seq_len, num_heads), dtype=torch.float32)).to(dtype)
    initial_state = torch.randn(
        (batch_size, num_heads, value_dim, key_dim),
        dtype=torch.float32,
    ).to(dtype)
    return q, k, v, g, beta, initial_state
