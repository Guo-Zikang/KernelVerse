from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.flash_mla_sparse.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Sparse MLA-style attention over selected key/value indices."""

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
        kv: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_compute = q.to(self.compute_dtype)
        kv_compute = kv.to(self.compute_dtype)
        seq_q, num_heads, _ = q_compute.shape
        out = torch.empty_like(q_compute)
        max_logits = torch.empty((seq_q, num_heads), dtype=torch.float32, device=q.device)
        lse = torch.empty((seq_q, num_heads), dtype=torch.float32, device=q.device)

        for query_idx in range(seq_q):
            for head_idx in range(num_heads):
                selected = indices[query_idx, head_idx].long()
                keys = kv_compute[selected, head_idx]
                scores = (keys * q_compute[query_idx, head_idx]).sum(dim=-1)
                scores = scores * self.softmax_scale
                probs = F.softmax(scores, dim=-1)
                out[query_idx, head_idx] = (probs[:, None] * keys).sum(dim=0)
                scores_fp32 = scores.to(torch.float32)
                max_logits[query_idx, head_idx] = scores_fp32.max()
                lse[query_idx, head_idx] = torch.logsumexp(scores_fp32, dim=0)

        return out.to(self.output_dtype), max_logits, lse


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    seq_q = int(shape["seq_q"])
    seq_kv = int(shape["seq_kv"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    topk = int(shape["topk"])
    dtype = _resolve_dtype(shape["dtype"])

    q = torch.randn((seq_q, num_heads, head_dim), dtype=torch.float32).to(dtype)
    kv = torch.randn((seq_kv, num_heads, head_dim), dtype=torch.float32).to(dtype)
    indices = torch.randint(0, seq_kv, (seq_q, num_heads, topk), dtype=torch.int32)
    return q, kv, indices
