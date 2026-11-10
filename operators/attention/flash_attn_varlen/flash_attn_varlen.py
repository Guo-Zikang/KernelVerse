from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.flash_attn_varlen.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Packed variable-length scaled dot-product attention returning output and LSE."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_dim = int(shape["head_dim"])
        self.softmax_scale = float(shape.get("softmax_scale", self.head_dim ** -0.5))
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
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_compute = q.to(self.compute_dtype)
        k_compute = k.to(self.compute_dtype)
        v_compute = v.to(self.compute_dtype)
        outputs = []
        lse_parts = []

        for batch_idx in range(cu_seqlens_q.numel() - 1):
            q_start = int(cu_seqlens_q[batch_idx].item())
            q_end = int(cu_seqlens_q[batch_idx + 1].item())
            k_start = int(cu_seqlens_k[batch_idx].item())
            k_end = int(cu_seqlens_k[batch_idx + 1].item())

            q_slice = q_compute[q_start:q_end]
            k_slice = k_compute[k_start:k_end]
            v_slice = v_compute[k_start:k_end]
            scores = torch.einsum("qhd,khd->hqk", q_slice, k_slice)
            scores = scores * self.softmax_scale
            probs = F.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", probs, v_slice))
            lse_parts.append(torch.logsumexp(scores.to(torch.float32), dim=-1))

        output = torch.cat(outputs, dim=0).to(self.output_dtype)
        softmax_lse = torch.cat(lse_parts, dim=-1).to(torch.float32)
        return output, softmax_lse


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    seq_lens = [int(seq_len) for seq_len in shape["seq_lens"]]
    total_tokens = sum(seq_lens)
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])

    dims = (total_tokens, num_heads, head_dim)
    q = torch.randn(dims, dtype=torch.float32).to(dtype)
    k = torch.randn(dims, dtype=torch.float32).to(dtype)
    v = torch.randn(dims, dtype=torch.float32).to(dtype)

    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + seq_len)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    return q, k, v, cu_seqlens, cu_seqlens.clone()
