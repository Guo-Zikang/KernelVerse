from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.concat_and_cache_mla.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Concatenates MLA latent KV and RoPE key features into the KV cache."""

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

    def forward(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        to_cache = torch.cat(
            [kv_c.to(self.compute_dtype), k_pe.to(self.compute_dtype)],
            dim=-1,
        ).to(kv_cache.dtype)

        cached = kv_cache.clone()
        flat_cache = cached.view(-1, *cached.shape[2:])
        valid = slot_mapping >= 0
        flat_cache[slot_mapping[valid].to(torch.long)] = to_cache[valid].to(flat_cache.dtype)
        return cached.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = int(shape["num_tokens"])
    num_blocks = int(shape["num_blocks"])
    block_size = int(shape["block_size"])
    latent_dim = int(shape["latent_dim"])
    rope_dim = int(shape["rope_dim"])

    kv_c = torch.randn((num_tokens, latent_dim), dtype=torch.float32).to(dtype)
    k_pe = torch.randn((num_tokens, rope_dim), dtype=torch.float32).to(dtype)
    kv_cache = torch.zeros((num_blocks, block_size, latent_dim + rope_dim), dtype=dtype)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64) % (num_blocks * block_size)
    return kv_c, k_pe, kv_cache, slot_mapping
