from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.static_sink_kv_populate.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Populate static sink key/value entries into a KV cache."""

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
        sink_key: torch.Tensor,
        sink_value: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        config: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block_size = int(config[0].item())
        sink_len = int(config[1].item())
        out_k = kv_cache_k.to(self.compute_dtype).clone()
        out_v = kv_cache_v.to(self.compute_dtype).clone()
        sink_key_compute = sink_key.to(self.compute_dtype)
        sink_value_compute = sink_value.to(self.compute_dtype)

        slot_mapping = torch.arange(
            block_size,
            block_size + sink_len,
            device=kv_cache_k.device,
        )
        out_k[slot_mapping.long()] = sink_key_compute
        out_v[slot_mapping.long()] = sink_value_compute
        return out_k.to(self.output_dtype), out_v.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    sink_len = int(shape["sink_len"])
    block_size = int(shape["block_size"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    total_slots = int(shape["total_slots"])

    sink_shape = (sink_len, num_kv_heads, head_dim)
    cache_shape = (total_slots, num_kv_heads, head_dim)
    sink_key = torch.randn(sink_shape, dtype=torch.float32).to(dtype)
    sink_value = torch.randn(sink_shape, dtype=torch.float32).to(dtype)
    kv_cache_k = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    kv_cache_v = torch.randn(cache_shape, dtype=torch.float32).to(dtype)
    config = torch.tensor([block_size, sink_len], dtype=torch.int32)
    return sink_key, sink_value, kv_cache_k, kv_cache_v, config
