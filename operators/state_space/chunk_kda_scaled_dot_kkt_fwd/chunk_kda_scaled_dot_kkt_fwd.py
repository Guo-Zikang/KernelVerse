from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.chunk_kda_scaled_dot_kkt_fwd.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Computes chunk-local KDA scaled KKT products and their cumulative sums."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
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
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, heads, _ = k.shape
        k_compute = k.to(self.compute_dtype)
        beta_compute = beta.to(self.compute_dtype)
        g_compute = g.to(self.compute_dtype)
        A = torch.zeros(
            (batch, time, heads, self.chunk_size),
            dtype=self.compute_dtype,
            device=k.device,
        )

        for start in range(0, time, self.chunk_size):
            end = min(time, start + self.chunk_size)
            cur = k_compute[:, start:end] * torch.exp(g_compute[:, start:end]).unsqueeze(-1)
            cur_beta = beta_compute[:, start:end].unsqueeze(-1)
            gram = torch.einsum("bthd,bshd->bhts", cur * cur_beta, cur)
            A[:, start:end, :, : end - start] = gram.permute(0, 2, 1, 3)

        Aqk = A.cumsum(dim=-1)
        return A.to(self.output_dtype), Aqk.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    dims = (
        int(shape["batch_size"]),
        int(shape["seq_len"]),
        int(shape["num_heads"]),
        int(shape["key_dim"]),
    )
    k = torch.randn(dims, dtype=torch.float32).to(dtype)
    beta = torch.sigmoid(torch.randn(dims[:-1], dtype=torch.float32)).to(dtype)
    g = torch.randn(dims[:-1], dtype=torch.float32).to(dtype)
    return k, beta, g
