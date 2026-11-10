from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.moe.masked_silu_and_mul.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference masked MoE SiLU gate activation."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, x: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
        num_experts, max_tokens, hidden_twice = x.shape
        hidden_dim = hidden_twice // 2
        out = torch.zeros(
            (num_experts, max_tokens, hidden_dim),
            dtype=self.output_dtype,
            device=x.device,
        )

        for expert_idx in range(num_experts):
            valid_tokens = max(0, min(int(token_counts[expert_idx].item()), max_tokens))
            if valid_tokens == 0:
                continue

            expert_input = x[expert_idx, :valid_tokens].to(self.compute_dtype)
            gate, up = torch.chunk(expert_input, 2, dim=-1)
            out[expert_idx, :valid_tokens] = (F.silu(gate) * up).to(self.output_dtype)

        return out


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_experts = int(shape["num_experts"])
    max_tokens = int(shape["max_tokens"])
    hidden_dim = int(shape["hidden_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    pattern = str(shape["token_count_pattern"])

    x = torch.randn(
        (num_experts, max_tokens, hidden_dim * 2),
        dtype=torch.float32,
    ).to(dtype)

    if pattern == "full":
        token_counts = torch.full((num_experts,), max_tokens, dtype=torch.int32)
    elif pattern == "sparse":
        high = max(1, max_tokens // 4)
        token_counts = torch.randint(0, high + 1, (num_experts,), dtype=torch.int32)
    else:
        token_counts = torch.randint(0, max_tokens + 1, (num_experts,), dtype=torch.int32)

    return x, token_counts
