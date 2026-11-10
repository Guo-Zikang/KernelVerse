from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.logits_processor_scale_softcap.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply soft capping and scaling to output logits."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.scale = float(shape.get("scale", 1.0))
        soft_cap = shape.get("soft_cap")
        self.soft_cap = None if soft_cap is None else float(soft_cap)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        out = logits.to(self.compute_dtype)

        if self.soft_cap is not None:
            soft_cap = torch.tensor(
                self.soft_cap,
                dtype=self.compute_dtype,
                device=logits.device,
            )
            out = torch.tanh(out / soft_cap) * soft_cap
        if self.scale != 1.0:
            out = out * self.scale
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    num_seqs = int(shape["num_seqs"])
    vocab_size = int(shape["vocab_size"])
    dtype = _resolve_dtype(shape["dtype"])
    logits = torch.randn((num_seqs, vocab_size), dtype=torch.float32).to(dtype)
    return (logits,)
