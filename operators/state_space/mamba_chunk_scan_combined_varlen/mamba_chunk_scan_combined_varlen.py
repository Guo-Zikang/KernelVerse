from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.state_space.mamba_chunk_scan_combined_varlen.helpers import (
    _resolve_dtype,
    validate_shape,
)


def _make_cu_seqlens(seq_lens: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    prefix = [0]
    running = 0
    for seq_len in seq_lens:
        running += int(seq_len)
        prefix.append(running)
    return torch.tensor(prefix, dtype=torch.int32, device=device)


def _build_chunk_metadata(
    seq_lens: Tuple[int, ...],
    chunk_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_chunk_seqlens = [0]
    seq_idx = []
    last_chunk_indices = []
    running_tokens = 0
    chunk_idx = -1

    for seq_id, seq_len in enumerate(seq_lens):
        remaining = int(seq_len)
        while remaining > 0:
            cur_len = min(chunk_size, remaining)
            running_tokens += cur_len
            cu_chunk_seqlens.append(running_tokens)
            seq_idx.append(seq_id)
            remaining -= cur_len
            chunk_idx += 1
        last_chunk_indices.append(chunk_idx)

    return (
        torch.tensor(cu_chunk_seqlens, dtype=torch.int32, device=device),
        torch.tensor(seq_idx, dtype=torch.int32, device=device),
        torch.tensor(last_chunk_indices, dtype=torch.int32, device=device),
    )


class Model(nn.Module):
    """Runs a variable-length chunked Mamba selective state-space scan."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
        self.dt_softplus = bool(shape["dt_softplus"])
        self.has_dt_bias = bool(shape["has_dt_bias"])
        self.has_D = bool(shape["has_D"])
        self.has_z = bool(shape["has_z"])
        self.has_initial_states = bool(shape["has_initial_states"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def _prepare_dt_chunks(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, nheads = dt.shape
        nchunks = cu_chunk_seqlens.numel() - 1
        dt_chunks = torch.zeros(
            (nchunks, nheads, self.chunk_size),
            dtype=self.compute_dtype,
            device=dt.device,
        )
        dA_cumsum = torch.zeros_like(dt_chunks).transpose(0, 1)
        A_compute = A.to(self.compute_dtype)

        for chunk_id in range(nchunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            cur_dt = dt[start:end].to(self.compute_dtype).transpose(0, 1)

            if dt_bias is not None:
                cur_dt = cur_dt + dt_bias.to(self.compute_dtype).view(-1, 1)
            if self.dt_softplus:
                cur_dt = torch.where(cur_dt <= 20.0, F.softplus(cur_dt), cur_dt)

            cur_dt = torch.clamp(cur_dt, min=0.0, max=float("inf"))
            dt_chunks[chunk_id, :, :chunk_len] = cur_dt
            dA_cumsum[:, chunk_id, :chunk_len] = torch.cumsum(
                cur_dt * A_compute.view(-1, 1),
                dim=-1,
            )

        return dA_cumsum, dt_chunks

    def _compute_chunk_states(
        self,
        x: torch.Tensor,
        B: torch.Tensor,
        dt_chunks: torch.Tensor,
        dA_cumsum: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        _, ngroups, dstate = B.shape
        _, nheads, hdim = x.shape
        nchunks = cu_chunk_seqlens.numel() - 1
        ratio = nheads // ngroups
        x_compute = x.to(self.compute_dtype)
        B_compute = B.to(self.compute_dtype)
        states = torch.zeros(
            (nchunks, nheads, hdim, dstate),
            dtype=self.compute_dtype,
            device=x.device,
        )

        for chunk_id in range(nchunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start

            for head in range(nheads):
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

    def _pass_states_between_chunks(
        self,
        states: torch.Tensor,
        dA_cumsum: torch.Tensor,
        last_chunk_indices: torch.Tensor,
        initial_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        states_flat = states.reshape(states.shape[0], states.shape[1], -1)
        init_flat = None
        if initial_states is not None:
            init_flat = initial_states.to(self.compute_dtype).reshape(
                initial_states.shape[0],
                initial_states.shape[1],
                -1,
            )

        nchunks, nheads, dim = states_flat.shape
        passed = torch.empty_like(states_flat, dtype=self.compute_dtype)
        chunk_start = 0

        for batch_id, chunk_end_tensor in enumerate(last_chunk_indices.to(torch.int64)):
            chunk_end = int(chunk_end_tensor.item()) + 1
            if init_flat is None:
                prev = torch.zeros(
                    (nheads, dim),
                    dtype=self.compute_dtype,
                    device=states.device,
                )
            else:
                prev = init_flat[batch_id].clone()

            for chunk_id in range(chunk_start, chunk_end):
                decay = torch.exp(dA_cumsum[:, chunk_id, -1]).unsqueeze(-1)
                prev = decay * prev + states_flat[chunk_id]
                passed[chunk_id] = prev
            chunk_start = chunk_end

        return passed.reshape_as(states)

    def _compute_chunk_gram(
        self,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        _, ngroups, _ = C.shape
        nchunks = cu_chunk_seqlens.numel() - 1
        B_compute = B.to(self.compute_dtype)
        C_compute = C.to(self.compute_dtype)
        gram_blocks = torch.zeros(
            (nchunks, ngroups, self.chunk_size, self.chunk_size),
            dtype=self.compute_dtype,
            device=C.device,
        )

        for chunk_id in range(nchunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            b_chunk = B_compute[start:end]
            c_chunk = C_compute[start:end]

            for group in range(ngroups):
                gram_blocks[chunk_id, group, :chunk_len, :chunk_len] = (
                    c_chunk[:, group] @ b_chunk[:, group].transpose(0, 1)
                )

        return gram_blocks

    def _compute_output(
        self,
        x: torch.Tensor,
        C: torch.Tensor,
        states: torch.Tensor,
        gram_blocks: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seqlen, nheads, hdim = x.shape
        _, ngroups, _ = C.shape
        ratio = nheads // ngroups
        nchunks = cu_chunk_seqlens.numel() - 1
        x_compute = x.to(self.compute_dtype)
        C_compute = C.to(self.compute_dtype)
        out = torch.zeros(
            (seqlen, nheads, hdim),
            dtype=self.compute_dtype,
            device=x.device,
        )

        for chunk_id in range(nchunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start

            for head in range(nheads):
                group = head // ratio
                x_chunk = x_compute[start:end, head]
                local_scores = gram_blocks[chunk_id, group, :chunk_len, :chunk_len]
                local_out = local_scores @ x_chunk
                state_out = torch.einsum(
                    "md,td->tm",
                    states[chunk_id, head],
                    C_compute[start:end, group],
                )
                total = local_out + state_out

                if D is not None:
                    d_term = D[head].to(self.compute_dtype)
                    if D.dim() != 2:
                        d_term = d_term.expand(hdim)
                    total = total + x_chunk * d_term
                if z is not None:
                    total = total * F.silu(z[start:end, head].to(self.compute_dtype))

                out[start:end, head] = total

        return out

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        last_chunk_indices: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cu_seqlens

        D_arg = D if self.has_D else None
        z_arg = z if self.has_z else None
        dt_bias_arg = dt_bias if self.has_dt_bias else None
        initial_states_arg = initial_states if self.has_initial_states else None

        dA_cumsum, dt_chunks = self._prepare_dt_chunks(
            dt,
            A,
            cu_chunk_seqlens,
            dt_bias_arg,
        )
        chunk_states = self._compute_chunk_states(
            x,
            B,
            dt_chunks,
            dA_cumsum,
            cu_chunk_seqlens,
        )
        passed_states = self._pass_states_between_chunks(
            chunk_states,
            dA_cumsum,
            last_chunk_indices,
            initial_states_arg,
        )
        gram_blocks = self._compute_chunk_gram(B, C, cu_chunk_seqlens)
        out = self._compute_output(
            x,
            C,
            passed_states,
            gram_blocks,
            cu_chunk_seqlens,
            D_arg,
            z_arg,
        )
        final_states = passed_states[last_chunk_indices.to(torch.int64)]

        return out.to(self.output_dtype), final_states.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    seq_lens = tuple(int(seq_len) for seq_len in shape["seq_lens"])
    total_len = sum(seq_lens)
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
    cu_seqlens = _make_cu_seqlens(seq_lens, x.device)
    cu_chunk_seqlens, _, last_chunk_indices = _build_chunk_metadata(
        seq_lens,
        chunk_size,
        x.device,
    )
    D = torch.randn((num_heads, head_dim), dtype=torch.float32).to(dtype)
    z = torch.randn((total_len, num_heads, head_dim), dtype=torch.float32).to(dtype)
    dt_bias = torch.randn((num_heads,), dtype=torch.float32).to(dtype)
    initial_states = torch.randn(
        (len(seq_lens), num_heads, head_dim, dstate),
        dtype=torch.float32,
    ).to(dtype)

    return (
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens,
        cu_chunk_seqlens,
        last_chunk_indices,
        D,
        z,
        dt_bias,
        initial_states,
    )
