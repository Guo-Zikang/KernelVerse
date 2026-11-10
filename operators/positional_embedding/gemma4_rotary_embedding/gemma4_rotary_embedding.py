from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.gemma4_rotary_embedding.helpers import (
    DEFAULT_BASE,
    DEFAULT_IS_NEOX_STYLE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply Gemma4 rotary embeddings to query and key tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_size = int(shape["head_size"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.base = float(shape.get("base", DEFAULT_BASE))
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
        inv_freq = self._gemma4_inv_freq(query.device)
        positions_compute = positions.to(device=inv_freq.device, dtype=self.compute_dtype)
        freqs = torch.einsum("i,j->ij", positions_compute, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        query_out = self._apply_rotary_embedding(query_compute, cos, sin)
        key_out = self._apply_rotary_embedding(key_compute, cos, sin)
        return query_out.to(self.output_dtype), key_out.to(self.output_dtype)

    def _gemma4_inv_freq(self, device: torch.device) -> torch.Tensor:
        rope_angles = self.rotary_dim // 2
        nope_angles = self.head_size // 2 - rope_angles
        freq_exponents = (
            torch.arange(
                0,
                2 * rope_angles,
                2,
                dtype=self.compute_dtype,
                device=device,
            )
            / self.head_size
        )
        inv_freq = 1.0 / self.base ** freq_exponents
        if nope_angles > 0:
            inv_freq = torch.cat(
                (
                    inv_freq,
                    torch.zeros(nope_angles, dtype=self.compute_dtype, device=device),
                ),
                dim=0,
            )
        return inv_freq

    def _apply_rotary_embedding(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = tensor.shape[0]
        tensor_view = tensor.view(num_tokens, -1, self.head_size)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        if self.is_neox_style:
            tensor_1, tensor_2 = torch.chunk(tensor_view, 2, dim=-1)
            output = torch.cat(
                (
                    tensor_1 * cos - tensor_2 * sin,
                    tensor_2 * cos + tensor_1 * sin,
                ),
                dim=-1,
            )
        else:
            tensor_1 = tensor_view[..., ::2]
            tensor_2 = tensor_view[..., 1::2]
            output = torch.stack(
                (
                    tensor_1 * cos - tensor_2 * sin,
                    tensor_2 * cos + tensor_1 * sin,
                ),
                dim=-1,
            ).flatten(-2)

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
