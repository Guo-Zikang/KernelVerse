import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.llama3_rotary_embedding.helpers import (
    DEFAULT_BASE,
    DEFAULT_IS_NEOX_STYLE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply Llama 3 scaled rotary embeddings to query and key tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_size = int(shape["head_size"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.base = float(shape.get("base", DEFAULT_BASE))
        self.scaling_factor = float(shape["scaling_factor"])
        self.low_freq_factor = float(shape["low_freq_factor"])
        self.high_freq_factor = float(shape["high_freq_factor"])
        self.original_max_position_embeddings = int(shape["original_max_position_embeddings"])
        self.is_neox_style = bool(shape.get("is_neox_style", DEFAULT_IS_NEOX_STYLE))
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
        inv_freq = self._llama3_inv_freq(query.device)
        cos, sin = self._build_cos_sin(positions, inv_freq)

        query_out = self._apply_rotary_embedding(query_compute, cos, sin)
        key_out = self._apply_rotary_embedding(key_compute, cos, sin)
        return query_out.to(self.output_dtype), key_out.to(self.output_dtype)

    def _llama3_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    dtype=self.compute_dtype,
                    device=device,
                )
                / self.rotary_dim
            )
        )
        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
        high_freq_wavelen = self.original_max_position_embeddings / self.high_freq_factor
        wave_len = 2.0 * math.pi / inv_freq
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (
                self.original_max_position_embeddings / wave_len
                - self.low_freq_factor
            ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = torch.zeros_like(wave_len)
        return torch.where(
            wave_len < high_freq_wavelen,
            inv_freq,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freq / self.scaling_factor,
                (1.0 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
            ),
        )

    def _build_cos_sin(
        self,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions_compute = positions.to(device=inv_freq.device, dtype=self.compute_dtype)
        freqs = torch.einsum("i,j->ij", positions_compute, inv_freq)
        return freqs.cos(), freqs.sin()

    def _apply_rotary_embedding(
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

        if self.is_neox_style:
            tensor_1, tensor_2 = torch.chunk(tensor_rot, 2, dim=-1)
            out_rot = torch.cat(
                (
                    tensor_1 * cos - tensor_2 * sin,
                    tensor_2 * cos + tensor_1 * sin,
                ),
                dim=-1,
            )
        else:
            tensor_1 = tensor_rot[..., ::2]
            tensor_2 = tensor_rot[..., 1::2]
            out_rot = torch.stack(
                (
                    tensor_1 * cos - tensor_2 * sin,
                    tensor_2 * cos + tensor_1 * sin,
                ),
                dim=-1,
            ).flatten(-2)

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
