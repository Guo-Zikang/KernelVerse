from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.windowed_topk.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Select top-k absolute indices from shifted valid score windows."""

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
        row_starts: torch.Tensor,
    ) -> torch.Tensor:
        score_compute = score.to(self.compute_dtype)
        rows = []
        for row in range(score_compute.shape[0]):
            start = int(row_starts[row].item())
            length = int(lengths[row].item())
            values = score_compute[row, start : start + length]
            indices = torch.argsort(values, descending=True, stable=True)[: self.topk].to(torch.int32) + start
            rows.append(indices)
        return torch.stack(rows, dim=0)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    max_start = length - window_length
    row_starts = torch.randint(
        0,
        max_start + 1,
        (batch,),
        dtype=torch.int32,
    )
    return score, lengths, row_starts
