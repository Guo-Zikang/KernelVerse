from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.state_space.ssd_chunk_scan.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Run the SSD chunk scan reference pipeline."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
        self.dt_softplus = bool(shape["dt_softplus"])
        self.has_dt_bias = bool(shape["has_dt_bias"])
        self.has_D = bool(shape["has_D"])
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
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        last_chunk_indices: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        dt_bias = dt_bias if self.has_dt_bias else None
        D = D if self.has_D else None
        z = z if self.has_z else None

        dt_chunks, dA_cumsum = self._prepare_delta(dt, A, cu_chunk_seqlens, dt_bias)
        chunk_states = self._compute_chunk_states(B, x, dt_chunks, dA_cumsum, cu_chunk_seqlens)
        passed_states = self._pass_states(chunk_states, dA_cumsum, last_chunk_indices)
        local_cb = self._chunk_bmm(C, B, cu_chunk_seqlens, causal=False)
        out = self._combine_outputs(
            x,
            C,
            local_cb,
            passed_states,
            cu_chunk_seqlens,
            D,
            z,
        )
        return out.to(self.output_dtype)

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
        dt_chunks: torch.Tensor,
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
                        decay * dt_chunks[chunk_id, head, pos] * (x_t @ b_t)
                    )
        return states

    def _pass_states(
        self,
        chunk_states: torch.Tensor,
        dA_cumsum: torch.Tensor,
        last_chunk_indices: torch.Tensor,
    ) -> torch.Tensor:
        flat_states = chunk_states.reshape(chunk_states.shape[0], chunk_states.shape[1], -1)
        num_chunks, num_heads, dim = flat_states.shape
        out = torch.empty_like(flat_states)
        chunk_start = 0

        for chunk_end_tensor in last_chunk_indices.to(torch.int64):
            chunk_end = int(chunk_end_tensor.item()) + 1
            prev = torch.zeros(
                (num_heads, dim),
                dtype=self.compute_dtype,
                device=flat_states.device,
            )
            for chunk_id in range(chunk_start, chunk_end):
                decay = torch.exp(dA_cumsum[:, chunk_id, -1]).unsqueeze(-1)
                prev = decay * prev + flat_states[chunk_id]
                out[chunk_id] = prev
            chunk_start = chunk_end

        if chunk_start < num_chunks:
            out[chunk_start:] = flat_states[chunk_start:]
        return out.reshape_as(chunk_states)

    def _chunk_bmm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        _, num_groups, _ = a.shape
        num_chunks = cu_chunk_seqlens.numel() - 1
        out = torch.zeros(
            (num_chunks, num_groups, self.chunk_size, self.chunk_size),
            dtype=self.compute_dtype,
            device=a.device,
        )
        a_compute = a.to(self.compute_dtype)
        b_compute = b.to(self.compute_dtype)

        for chunk_id in range(num_chunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            a_chunk = a_compute[start:end]
            b_chunk = b_compute[start:end]
            for group in range(num_groups):
                gram = a_chunk[:, group] @ b_chunk[:, group].transpose(0, 1)
                if causal:
                    mask = torch.tril(torch.ones_like(gram, dtype=torch.bool))
                    gram = torch.where(mask, gram, torch.zeros_like(gram))
                out[chunk_id, group, :chunk_len, :chunk_len] = gram
        return out

    def _combine_outputs(
        self,
        x: torch.Tensor,
        C: torch.Tensor,
        cb: torch.Tensor,
        states: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seq_len, num_heads, head_dim = x.shape
        _, num_groups, _ = C.shape
        ratio = num_heads // num_groups
        out = torch.zeros(
            (seq_len, num_heads, head_dim),
            dtype=self.compute_dtype,
            device=x.device,
        )
        num_chunks = cu_chunk_seqlens.numel() - 1
        x_compute = x.to(self.compute_dtype)
        C_compute = C.to(self.compute_dtype)
        D_compute = None if D is None else D.to(self.compute_dtype)
        z_compute = None if z is None else z.to(self.compute_dtype)

        for chunk_id in range(num_chunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            for head in range(num_heads):
                group = head // ratio
                x_chunk = x_compute[start:end, head]
                local_scores = cb[chunk_id, group, :chunk_len, :chunk_len]
                local_out = local_scores @ x_chunk
                state_out = torch.einsum(
                    "md,td->tm",
                    states[chunk_id, head],
                    C_compute[start:end, group],
                )
                total = local_out + state_out
                if D_compute is not None:
                    total = total + x_chunk * D_compute[head]
                if z_compute is not None:
                    z_chunk = z_compute[start:end, head]
                    total = total * (z_chunk * torch.sigmoid(z_chunk))
                out[start:end, head] = total
        return out


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    total_len = sum((int(seq_len) for seq_len in shape["seq_lens"]))
    num_heads = int(shape["num_heads"])
    num_groups = int(shape["num_groups"])
    head_dim = int(shape["head_dim"])
    dstate = int(shape["dstate"])
    chunk_size = int(shape["chunk_size"])

    x = torch.randn((total_len, num_heads, head_dim), dtype=torch.float32).to(dtype)
    dt = torch.randn((total_len, num_heads), dtype=torch.float32).to(dtype)
    A = -torch.rand((num_heads,), dtype=torch.float32).to(dtype)
    B = torch.randn((total_len, num_groups, dstate), dtype=torch.float32).to(dtype)
    C = torch.randn((total_len, num_groups, dstate), dtype=torch.float32).to(dtype)

    cu_chunk_seqlens = [0]
    last_chunk_indices = []
    running_tokens = 0
    chunk_index = -1
    for seq_len in shape["seq_lens"]:
        remaining = int(seq_len)
        while remaining > 0:
            cur = min(chunk_size, remaining)
            running_tokens += cur
            cu_chunk_seqlens.append(running_tokens)
            remaining -= cur
            chunk_index += 1
        last_chunk_indices.append(chunk_index)
    cu_chunk_seqlens_tensor = torch.tensor(cu_chunk_seqlens, dtype=torch.int32)
    last_chunk_indices_tensor = torch.tensor(last_chunk_indices, dtype=torch.int32)

    D = torch.randn((num_heads, head_dim), dtype=torch.float32).to(dtype)
    z = torch.randn((total_len, num_heads, head_dim), dtype=torch.float32).to(dtype)
    dt_bias = torch.randn((num_heads,), dtype=torch.float32).to(dtype)
    return (
        x,
        dt,
        A,
        B,
        C,
        cu_chunk_seqlens_tensor,
        last_chunk_indices_tensor,
        D,
        z,
        dt_bias,
    )
