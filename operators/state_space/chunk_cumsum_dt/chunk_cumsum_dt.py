from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.state_space.chunk_cumsum_dt.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Computes chunked dt values and dA cumulative sums for Mamba-style scans."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
        self.dt_softplus = bool(shape["dt_softplus"])
        self.has_dt_bias = bool(shape["has_dt_bias"])
        self.input_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.input_dtype
        )

    def _prepare_dt_chunk(
        self,
        dt_chunk: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cur = dt_chunk.to(self.compute_dtype).transpose(0, 1)
        if dt_bias is not None:
            cur = cur + dt_bias.to(self.compute_dtype).view(-1, 1)
        if self.dt_softplus:
            cur = torch.where(cur <= 20.0, F.softplus(cur), cur)
        return torch.clamp(cur, min=0.0, max=float("inf"))

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bias = dt_bias if self.has_dt_bias else None
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

        for chunk_id in range(num_chunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            cur = self._prepare_dt_chunk(dt[start:end], bias)
            dA = cur * A_compute.view(-1, 1)
            dt_out[chunk_id, :, :chunk_len] = cur
            dA_cumsum[:, chunk_id, :chunk_len] = torch.cumsum(dA, dim=-1)
        return dA_cumsum.to(self.input_dtype), dt_out.to(self.input_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    total_len = sum((int(seq_len) for seq_len in shape["seq_lens"]))
    num_heads = int(shape["num_heads"])
    chunk_size = int(shape["chunk_size"])

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
    return dt, A, cu_chunk_seqlens_tensor, dt_bias
