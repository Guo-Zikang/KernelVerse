from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.phi3_long_rope_scaled_rotary_embedding.helpers import (
    DEFAULT_BASE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply Phi-3 LongRoPE scaling to flattened query and key tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_size = int(shape["head_size"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.base = float(shape.get("base", DEFAULT_BASE))
        self.original_max_position_embeddings = int(shape["original_max_position_embeddings"])
        self.use_long_rope = bool(shape.get("use_long_rope", True))
        self.short_factor = list(shape["short_factor"])
        self.long_factor = list(shape["long_factor"])
        self.short_mscale = float(shape.get("short_mscale", 1.0))
        self.long_mscale = float(shape.get("long_mscale", 1.0))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        factors = self.long_factor if self.use_long_rope else self.short_factor
        mscale = self.long_mscale if self.use_long_rope else self.short_mscale
        cos, sin = self._build_cos_sin(positions, query.device, factors, mscale)

        query_out = self._apply_neox_rotary_embedding(query_compute, cos, sin)
        key_out = self._apply_neox_rotary_embedding(key_compute, cos, sin)
        return query_out.to(self.output_dtype), key_out.to(self.output_dtype)

    def _build_cos_sin(
        self,
        positions: torch.Tensor,
        device: torch.device,
        factors: Any,
        mscale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rescale_factors = torch.tensor(
            list(factors),
            device=device,
            dtype=self.compute_dtype,
        )
        inv_freq = 1.0 / (
            rescale_factors
            * self.base
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    device=device,
                    dtype=self.compute_dtype,
                )
                / self.rotary_dim
            )
        )
        freqs = torch.einsum(
            "i,j->ij",
            positions.to(device=device, dtype=self.compute_dtype),
            inv_freq,
        )
        return (
            freqs.cos() * mscale,
            freqs.sin() * mscale,
        )

    def _apply_neox_rotary_embedding(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = tensor.shape[0]
        tensor_view = tensor.view(num_tokens, -1, self.head_size)
        tensor_rot = tensor_view[..., : self.rotary_dim]
        tensor_pass = tensor_view[..., self.rotary_dim :]

        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        tensor_1, tensor_2 = torch.chunk(tensor_rot, 2, dim=-1)
        out_rot = torch.cat(
            (
                tensor_1 * cos - tensor_2 * sin,
                tensor_2 * cos + tensor_1 * sin,
            ),
            dim=-1,
        )
        output = torch.cat((out_rot, tensor_pass), dim=-1)
        return output.reshape(tensor.shape)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_size = int(shape["head_size"])
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = batch_size * seq_len
    positions = torch.arange(seq_len, dtype=torch.int32).repeat(batch_size)
    query = torch.randn(
        (num_tokens, num_q_heads * head_size),
        dtype=torch.float32,
    ).to(dtype)
    key = torch.randn(
        (num_tokens, num_kv_heads * head_size),
        dtype=torch.float32,
    ).to(dtype)
    return positions, query, key
