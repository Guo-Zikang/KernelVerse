from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.variable_length_topk.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Select top-k indices from each score row using per-row valid lengths."""

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

    def forward(self, score: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        score_compute = score.to(self.compute_dtype)
        outputs = []
        for row, length in zip(score_compute, lengths):
            valid_length = int(length.item())
            if valid_length <= self.topk:
                padded = torch.full(
                    (self.topk,),
                    -1,
                    dtype=torch.int32,
                    device=score.device,
                )
                padded[:valid_length] = torch.arange(
                    valid_length,
                    dtype=torch.int32,
                    device=score.device,
                )
            else:
                valid_scores = row[:valid_length]
                padded = torch.argsort(
                    valid_scores,
                    dim=-1,
                    descending=True,
                    stable=True,
                )[: self.topk].to(torch.int32)
            outputs.append(padded)
        return torch.stack(outputs, dim=0)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch = int(shape["batch"])
    length = int(shape["length"])
    topk = int(shape["topk"])
    dtype = _resolve_dtype(shape["dtype"])
    score = torch.randn(
        (batch, length),
        dtype=torch.float32,
    )
    lengths = torch.randint(1, length + 1, (batch,), dtype=torch.int32)
    boundary_lengths = [1, min(topk, length), min(topk + 1, length), length]
    for row, valid_length in enumerate(boundary_lengths[:batch]):
        lengths[row] = valid_length

    for row, valid_length in enumerate(lengths.tolist()):
        if valid_length < length:
            score[row, valid_length:] = 1000.0

    score = score.to(dtype)
    return score, lengths
