from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.mrotary_embedding.helpers import (
    DEFAULT_BASE,
    DEFAULT_IS_NEOX_STYLE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply multimodal rotary embeddings to query and key tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_size = int(shape["head_size"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.sections = [int(section) for section in shape["mrope_section"]]
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
        cos, sin = self._mrope_cos_sin(positions, query.device)
        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        query_out = self._apply_rotary_embedding(query_compute, cos, sin)
        key_out = self._apply_rotary_embedding(key_compute, cos, sin)
        return query_out.to(self.output_dtype), key_out.to(self.output_dtype)

    def _mrope_cos_sin(
        self,
        positions: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_parts = []
        sin_parts = []
        for index, section in enumerate(self.sections):
            inv_freq = self._build_inv_freq(section * 2, device)
            cos_i, sin_i = self._build_cos_sin(positions[index], inv_freq)
            cos_parts.append(cos_i)
            sin_parts.append(sin_i)
        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

    def _build_inv_freq(self, rotary_dim: int, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=self.compute_dtype,
                    device=device,
                )
                / rotary_dim
            )
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
    token_positions = torch.arange(seq_len, dtype=torch.int32).repeat(batch_size)
    height_positions = (torch.arange(seq_len, dtype=torch.int32) % 8).repeat(batch_size)
    width_positions = (torch.arange(seq_len, dtype=torch.int32) % 8).flip(0).repeat(batch_size)
    positions = torch.stack([token_positions, height_positions, width_positions], dim=0)
    query = torch.randn(
        (num_tokens, num_q_heads * head_size),
        dtype=torch.float32,
    ).to(dtype)
    key = torch.randn(
        (num_tokens, num_kv_heads * head_size),
        dtype=torch.float32,
    ).to(dtype)
    return positions, query, key
