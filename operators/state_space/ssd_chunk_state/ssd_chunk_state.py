from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.state_space.ssd_chunk_state.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Compute SSD chunk states from inputs, deltas, and B parameters."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
        self.dt_softplus = bool(shape["dt_softplus"])
        self.has_dt_bias = bool(shape["has_dt_bias"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        B: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        dt_bias = dt_bias if self.has_dt_bias else None
        dt_out, dA_cumsum = self._prepare_delta(dt, A, cu_chunk_seqlens, dt_bias)
        states = self._compute_chunk_states(B, x, dt_out, dA_cumsum, cu_chunk_seqlens)
        return states.to(self.output_dtype)

    def _prepare_delta(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, num_heads = dt.shape
        num_chunks = cu_chunk_seqlens.numel() - 1
        dt_out = torch.zeros(
            (num_chunks, num_heads, self.chunk_size),
            dtype=self.compute_dtype,
            device=dt.device,
        )
        dA_cumsum = torch.zeros(
            (num_heads, num_chunks, self.chunk_size),
            dtype=self.compute_dtype,
            device=dt.device,
        )
        A_compute = A.to(self.compute_dtype)
        dt_compute = dt.to(self.compute_dtype)

        for chunk_id in range(num_chunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            cur = dt_compute[start:end].transpose(0, 1)
            if dt_bias is not None:
                cur = cur + dt_bias.to(self.compute_dtype).view(-1, 1)
            if self.dt_softplus:
                cur = torch.where(cur <= 20.0, F.softplus(cur), cur)
            cur = torch.clamp(cur, min=0.0, max=float("inf"))
            dt_out[chunk_id, :, :chunk_len] = cur
            dA = cur * A_compute.view(-1, 1)
            dA_cumsum[:, chunk_id, :chunk_len] = torch.cumsum(dA, dim=-1)
        return dt_out, dA_cumsum

    def _compute_chunk_states(
        self,
        B: torch.Tensor,
        x: torch.Tensor,
        dt_out: torch.Tensor,
        dA_cumsum: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        _, num_groups, dstate = B.shape
        _, num_heads, head_dim = x.shape
        num_chunks = cu_chunk_seqlens.numel() - 1
        ratio = num_heads // num_groups
        states = torch.zeros(
            (num_chunks, num_heads, head_dim, dstate),
            dtype=self.compute_dtype,
            device=x.device,
        )
        B_compute = B.to(self.compute_dtype)
        x_compute = x.to(self.compute_dtype)

        for chunk_id in range(num_chunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            for head in range(num_heads):
                group = head // ratio
                last_decay = dA_cumsum[head, chunk_id, chunk_len - 1]
                for pos in range(chunk_len):
                    decay = torch.exp(last_decay - dA_cumsum[head, chunk_id, pos])
                    x_t = x_compute[start + pos, head].unsqueeze(-1)
                    b_t = B_compute[start + pos, group].unsqueeze(0)
                    states[chunk_id, head] += (
                        decay * dt_out[chunk_id, head, pos] * (x_t @ b_t)
                    )
        return states


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    total_len = sum((int(seq_len) for seq_len in shape["seq_lens"]))
    num_heads = int(shape["num_heads"])
    num_groups = int(shape["num_groups"])
    head_dim = int(shape["head_dim"])
    dstate = int(shape["dstate"])
    chunk_size = int(shape["chunk_size"])

    B = torch.randn((total_len, num_groups, dstate), dtype=torch.float32).to(dtype)
    x = torch.randn((total_len, num_heads, head_dim), dtype=torch.float32).to(dtype)
    dt = torch.randn((total_len, num_heads), dtype=torch.float32).to(dtype)
    A = -torch.rand((num_heads,), dtype=torch.float32).to(dtype)

    cu_chunk_seqlens = [0]
    running_tokens = 0
    for seq_len in shape["seq_lens"]:
        remaining = int(seq_len)
        while remaining > 0:
            cur = min(chunk_size, remaining)
            running_tokens += cur
            cu_chunk_seqlens.append(running_tokens)
            remaining -= cur
    cu_chunk_seqlens_tensor = torch.tensor(cu_chunk_seqlens, dtype=torch.int32)

    dt_bias = torch.randn((num_heads,), dtype=torch.float32).to(dtype)
    return B, x, dt, A, cu_chunk_seqlens_tensor, dt_bias
