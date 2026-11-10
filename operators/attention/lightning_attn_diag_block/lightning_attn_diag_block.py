from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.lightning_attn_diag_block.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Compute decayed intra-block lightning attention outputs."""

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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        block_size = self.block_size
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        s_compute = s.to(torch.float32)
        (batch_size, num_heads, seq_len, _) = q_compute.shape
        out = torch.zeros(
            (batch_size, num_heads, seq_len, v_compute.shape[-1]),
            dtype=self.compute_dtype,
            device=q.device,
        )

        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                decay = s_compute[head_idx]
                for start in range(0, seq_len, block_size):
                    end = min(start + block_size, seq_len)
                    q_blk = q_compute[batch_idx, head_idx, start:end]
                    k_blk = k_compute[batch_idx, head_idx, start:end]
                    v_blk = v_compute[batch_idx, head_idx, start:end]

                    scores = q_blk @ k_blk.transpose(0, 1)
                    idx = torch.arange(end - start, device=q.device)
                    rel = idx[:, None] - idx[None, :]
                    mask = rel >= 0
                    decay_mask = torch.where(
                        mask,
                        torch.exp(-decay * rel.to(torch.float32)),
                        torch.zeros_like(rel, dtype=torch.float32),
                    )
                    weighted_scores = (scores.to(torch.float32) * decay_mask).to(
                        self.compute_dtype
                    )
                    out[batch_idx, head_idx, start:end] = weighted_scores @ v_blk
        return out.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    num_heads = int(shape["num_heads"])
    seq_len = int(shape["seq_len"])
    head_dim = int(shape["head_dim"])
    value_dim = int(shape["value_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    q = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=torch.float32,
    ).to(dtype)
    k = torch.randn_like(q)
    v = torch.randn(
        (batch_size, num_heads, seq_len, value_dim),
        dtype=torch.float32,
    ).to(dtype)
    s = torch.rand((num_heads,), dtype=torch.float32)
    return (q, k, v, s)
