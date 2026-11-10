from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.ssd_state_passing.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Pass SSD chunk states across sequence chunks with decay."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.has_initial_states = bool(shape["has_initial_states"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        states: torch.Tensor,
        dA_cumsum: torch.Tensor,
        last_chunk_indices: torch.Tensor,
        initial_states: torch.Tensor,
    ) -> torch.Tensor:
        initial_states = initial_states if self.has_initial_states else None
        num_chunks, num_heads, state_dim = states.shape
        states_compute = states.to(self.compute_dtype)
        dA_compute = dA_cumsum.to(self.compute_dtype)
        out = torch.empty_like(states_compute)
        chunk_start = 0

        for batch_id, chunk_end_tensor in enumerate(last_chunk_indices.to(torch.int64)):
            chunk_end = int(chunk_end_tensor.item()) + 1
            if initial_states is None:
                prev = torch.zeros(
                    (num_heads, state_dim),
                    dtype=self.compute_dtype,
                    device=states.device,
                )
            else:
                prev = initial_states[batch_id].to(self.compute_dtype).clone()
            for chunk_id in range(chunk_start, chunk_end):
                decay = torch.exp(dA_compute[:, chunk_id, -1]).unsqueeze(-1)
                prev = decay * prev + states_compute[chunk_id]
                out[chunk_id] = prev
            chunk_start = chunk_end

        if chunk_start < num_chunks:
            out[chunk_start:] = states_compute[chunk_start:]
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch_size = int(shape["batch_size"])
    num_chunks = int(shape["num_chunks"])
    num_heads = int(shape["num_heads"])
    state_dim = int(shape["state_dim"])
    chunk_size = int(shape["chunk_size"])

    states = torch.randn((num_chunks, num_heads, state_dim), dtype=torch.float32).to(dtype)
    dA_cumsum = torch.randn((num_heads, num_chunks, chunk_size), dtype=torch.float32).to(dtype)

    boundaries = []
    last = -1
    base = num_chunks // batch_size
    remainder = num_chunks % batch_size
    for batch_idx in range(batch_size):
        span = base + (1 if batch_idx < remainder else 0)
        last += span
        boundaries.append(last)
    last_chunk_indices = torch.tensor(boundaries, dtype=torch.int32)

    initial_states = torch.randn(
        (batch_size, num_heads, state_dim),
        dtype=torch.float32,
    ).to(dtype)
    return states, dA_cumsum, last_chunk_indices, initial_states
