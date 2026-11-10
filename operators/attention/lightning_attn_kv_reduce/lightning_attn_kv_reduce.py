from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.lightning_attn_kv_reduce.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reduce decayed key-value products into lightning attention blocks."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.block_size = int(shape["block_size"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_decay: torch.Tensor,
    ) -> torch.Tensor:
        block_size = self.block_size
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        k_decay_compute = k_decay.to(self.compute_dtype)
        (batch_size, num_heads, seq_len, head_dim) = k_compute.shape
        value_dim = v_compute.shape[-1]
        num_blocks = (seq_len + block_size - 1) // block_size
        out = torch.zeros(
            (batch_size, num_heads, num_blocks, head_dim, value_dim),
            dtype=self.compute_dtype,
            device=k.device,
        )

        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                for block_idx in range(num_blocks):
                    start = block_idx * block_size
                    end = min(start + block_size, seq_len)
                    k_blk = k_compute[batch_idx, head_idx, start:end]
                    v_blk = v_compute[batch_idx, head_idx, start:end]
                    decay = k_decay_compute[head_idx, : end - start].unsqueeze(-1)
                    out[batch_idx, head_idx, block_idx] = (k_blk * decay).transpose(
                        0,
                        1,
                    ) @ v_blk
        return out.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    num_heads = int(shape["num_heads"])
    seq_len = int(shape["seq_len"])
    head_dim = int(shape["head_dim"])
    value_dim = int(shape["value_dim"])
    block_size = int(shape["block_size"])
    dtype = _resolve_dtype(shape["dtype"])
    k = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    v = torch.randn(
        (batch_size, num_heads, seq_len, value_dim),
        dtype=torch.float32,
    ).to(dtype)
    k_decay = torch.rand((num_heads, block_size), dtype=torch.float32).to(dtype)
    return (k, v, k_decay)
