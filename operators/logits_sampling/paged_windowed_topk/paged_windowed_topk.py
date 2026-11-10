from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.paged_windowed_topk.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Select top-k score positions and map them through a page table."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.topk = int(shape["topk"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        page_table: torch.Tensor,
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        score_compute = score.to(self.compute_dtype)
        rows = []
        for row in range(score_compute.shape[0]):
            start = int(row_starts[row].item())
            length = int(lengths[row].item())
            values = score_compute[row, start : start + length]
            local_indices = torch.argsort(values, descending=True, stable=True)[: self.topk].long() + start
            rows.append(page_table[row, local_indices].to(torch.int32))
        return torch.stack(rows, dim=0)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch = int(shape["batch"])
    length = int(shape["length"])
    window_length = int(shape["window_length"])
    dtype = _resolve_dtype(shape["dtype"])

    score = torch.randn(
        (batch, length),
        dtype=torch.float32,
    ).to(dtype)
    lengths = torch.full(
        (batch,),
        window_length,
        dtype=torch.int32,
    )
    page_table = torch.arange(
        batch * length,
        dtype=torch.int32,
    ).view(batch, length)
    max_start = length - window_length
    row_starts = torch.randint(
        0,
        max_start + 1,
        (batch,),
        dtype=torch.int32,
    )
    return score, lengths, page_table, row_starts
