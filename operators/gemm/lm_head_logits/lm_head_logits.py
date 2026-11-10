from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.gemm.lm_head_logits.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference LM head vocabulary projection."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.use_bias = bool(shape["use_bias"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_compute = hidden_states.to(self.compute_dtype)
        weight_compute = weight.to(self.compute_dtype)
        bias_compute = None if not self.use_bias or bias is None else bias.to(self.compute_dtype)
        logits = F.linear(hidden_compute, weight_compute, bias_compute)
        return logits.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    hidden_dim = int(shape["hidden_dim"])
    vocab_size = int(shape["vocab_size"])
    dtype = _resolve_dtype(shape["dtype"])
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.float32).to(dtype)
    weight = torch.randn((vocab_size, hidden_dim), dtype=torch.float32).to(dtype)
    bias = (
        torch.randn((vocab_size,), dtype=torch.float32).to(dtype) * 0.01
        if bool(shape["use_bias"])
        else None
    )
    return hidden_states, weight, bias
