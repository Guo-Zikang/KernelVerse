from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.state_space.ssd_bmm.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Compute chunked state-space grouped Gram matrices."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.chunk_size = int(shape["chunk_size"])
        self.causal = bool(shape["causal"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        cu_chunk_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        _, num_groups, _ = a.shape
        num_chunks = cu_chunk_seqlens.numel() - 1
        out = torch.zeros(
            (num_chunks, num_groups, self.chunk_size, self.chunk_size),
            dtype=self.compute_dtype,
            device=a.device,
        )
        a_compute = a.to(self.compute_dtype)
        b_compute = b.to(self.compute_dtype)

        for chunk_id in range(num_chunks):
            start = int(cu_chunk_seqlens[chunk_id].item())
            end = int(cu_chunk_seqlens[chunk_id + 1].item())
            chunk_len = end - start
            a_chunk = a_compute[start:end]
            b_chunk = b_compute[start:end]
            for group in range(num_groups):
                gram = a_chunk[:, group] @ b_chunk[:, group].transpose(0, 1)
                if self.causal:
                    mask = torch.tril(torch.ones_like(gram, dtype=torch.bool))
                    gram = torch.where(mask, gram, torch.zeros_like(gram))
                out[chunk_id, group, :chunk_len, :chunk_len] = gram
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    total_len = sum((int(seq_len) for seq_len in shape["seq_lens"]))
    num_groups = int(shape["num_groups"])
    key_dim = int(shape["key_dim"])
    chunk_size = int(shape["chunk_size"])

    a = torch.randn((total_len, num_groups, key_dim), dtype=torch.float32).to(dtype)
    b = torch.randn((total_len, num_groups, key_dim), dtype=torch.float32).to(dtype)

    cu_chunk_seqlens = [0]
    running_tokens = 0
    for seq_len in shape["seq_lens"]:
        remaining = int(seq_len)
        while remaining > 0:
            cur = min(chunk_size, remaining)
            running_tokens += cur
            cu_chunk_seqlens.append(running_tokens)
            remaining -= cur
    cu_chunk_seqlens_tensor = torch.tensor(cu_chunk_seqlens, dtype=torch.int32)
    return a, b, cu_chunk_seqlens_tensor
