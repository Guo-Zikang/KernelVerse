from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.logits_sampling.sparse_decode_topk.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Select decode-time top-k sparse attention column indices."""

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

    def forward(self, logits: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        logits_compute = logits.to(self.compute_dtype)
        outputs = []
        for row in range(logits_compute.shape[0]):
            valid_length = int(seq_lens[row].item())
            valid = logits_compute[row, :valid_length]
            k = min(self.topk, valid.numel())
            idx = torch.argsort(valid, descending=True, stable=True)[:k]
            if idx.numel() < self.topk:
                idx = F.pad(idx, (0, self.topk - idx.numel()), value=-1)
            outputs.append(idx.to(torch.int32))
        return torch.stack(outputs, dim=0)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_rows = int(shape["num_rows"])
    num_cols = int(shape["num_cols"])
    topk = int(shape["topk"])
    logits = torch.randn((num_rows, num_cols), dtype=torch.float32).to(dtype)
    seq_lens = torch.randint(topk, num_cols + 1, (num_rows,), dtype=torch.int32)
    return logits, seq_lens
