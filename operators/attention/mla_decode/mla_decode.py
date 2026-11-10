from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.mla_decode.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference MLA decode operator with paged KV-cache semantics."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.latent_dim = 512
        self.rope_dim = 64
        self.total_dim = self.latent_dim + self.rope_dim
        self.sm_scale = 1.44
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_q_heads, _ = q_nope.shape
        output = torch.zeros(
            (batch_size, num_q_heads, self.latent_dim),
            dtype=self.compute_dtype,
            device=q_nope.device,
        )
        query = torch.cat(
            (q_nope.to(self.compute_dtype), q_pe.to(self.compute_dtype)),
            dim=-1,
        )
        kv_cache_compute = kv_cache.to(self.compute_dtype)

        for batch_idx in range(batch_size):
            seq_len = int(seq_lens[batch_idx].item())
            kv_blocks = kv_cache_compute[block_table[batch_idx].long()]
            kv_flat = kv_blocks.reshape(-1, self.total_dim)[:seq_len]
            key = kv_flat
            value = kv_flat[:, : self.latent_dim]

            logits = torch.matmul(query[batch_idx], key.transpose(0, 1)) * self.sm_scale
            probs = F.softmax(logits, dim=-1)
            output[batch_idx] = torch.matmul(probs, value)

        return output.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    latent_dim = 512
    rope_dim = 64
    total_dim = latent_dim + rope_dim
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_q_heads = int(shape["num_q_heads"])
    block_size = int(shape["block_size"])
    dtype = _resolve_dtype(shape["dtype"])
    block_num = (seq_len + block_size - 1) // block_size
    pack_factor = 128 // block_size
    block_num = (block_num + pack_factor - 1) // pack_factor * pack_factor

    q_nope = torch.randn(
        (batch_size, num_q_heads, latent_dim),
        dtype=torch.float32,
    ).to(dtype)
    q_pe = torch.randn(
        (batch_size, num_q_heads, rope_dim),
        dtype=torch.float32,
    ).to(dtype)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
    block_table = torch.randint(
        0,
        batch_size * block_num,
        (batch_size, block_num),
        dtype=torch.int32,
    )
    kv_cache = torch.randn(
        (block_table.numel(), block_size, total_dim),
        dtype=torch.float32,
    ).to(dtype)
    return q_nope, q_pe, kv_cache, seq_lens, block_table
