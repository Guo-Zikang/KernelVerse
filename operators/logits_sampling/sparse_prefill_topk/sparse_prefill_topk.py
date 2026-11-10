from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.logits_sampling.sparse_prefill_topk.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Select prefill-time top-k sparse attention column indices."""

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
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        logits_compute = logits.to(self.compute_dtype)
        outputs = []
        for row in range(logits_compute.shape[0]):
            start = int(cu_seqlen_ks[row].item())
            end = int(cu_seqlen_ke[row].item())
            valid = logits_compute[row, start:end]
            k = min(self.topk, valid.numel())
            idx = torch.argsort(valid, descending=True, stable=True)[:k] + start
            if idx.numel() < self.topk:
                idx = F.pad(idx, (0, self.topk - idx.numel()), value=-1)
            outputs.append(idx.to(torch.int32))
        return torch.stack(outputs, dim=0)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_rows = int(shape["num_rows"])
    num_cols = int(shape["num_cols"])
    range_len = int(shape["range_len"])

    logits = torch.randn((num_rows, num_cols), dtype=torch.float32).to(dtype)
    max_start = num_cols - range_len
    if max_start == 0:
        starts = torch.zeros((num_rows,), dtype=torch.int32)
    else:
        starts = torch.randint(0, max_start + 1, (num_rows,), dtype=torch.int32)
    ends = starts + range_len
    return logits, starts, ends
