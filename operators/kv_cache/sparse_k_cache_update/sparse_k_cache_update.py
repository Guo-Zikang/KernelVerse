from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.sparse_k_cache_update.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Update sparse attention key cache entries by slot mapping."""

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
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        out = kv_cache.to(self.compute_dtype).clone()
        k_compute = k.to(self.compute_dtype)
        valid = slot_mapping >= 0
        out[slot_mapping[valid].to(torch.long)] = k_compute[valid]
        return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = int(shape["num_tokens"])
    num_slots = int(shape["num_slots"])
    head_dim = int(shape["head_dim"])

    k = torch.randn((num_tokens, head_dim), dtype=torch.float32).to(dtype)
    slot_mapping = torch.randperm(num_slots, dtype=torch.int64)[:num_tokens].to(torch.int32)
    kv_cache = torch.randn((num_slots, head_dim), dtype=torch.float32).to(dtype)
    return k, slot_mapping, kv_cache
