from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.chunk_local_cumsum.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Computes chunk-local cumulative sums for scalar or vector gate tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
        self.reverse = bool(shape["reverse"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        g_compute = g.to(self.compute_dtype)
        out = torch.empty_like(g_compute)
        time = g_compute.shape[1]

        for start in range(0, time, self.chunk_size):
            end = min(time, start + self.chunk_size)
            block = g_compute[:, start:end]
            if self.reverse:
                block = torch.flip(torch.cumsum(torch.flip(block, dims=(1,)), dim=1), dims=(1,))
            else:
                block = torch.cumsum(block, dim=1)
            out[:, start:end] = block
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    dims = (
        int(shape["batch_size"]),
        int(shape["seq_len"]),
        int(shape["num_heads"]),
    )
    if "width" in shape:
        dims = (*dims, int(shape["width"]))
    g = torch.randn(dims, dtype=torch.float32).to(dtype)
    return (g,)
