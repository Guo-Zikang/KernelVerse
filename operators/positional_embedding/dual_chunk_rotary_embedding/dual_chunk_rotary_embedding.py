from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.dual_chunk_rotary_embedding.helpers import (
    DEFAULT_BASE,
    DEFAULT_IS_NEOX_STYLE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Build dual-chunk rotary query variants and rotary-encoded keys."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_size = int(shape["head_size"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.chunk_size = int(shape["chunk_size"])
        self.local_size = int(shape["local_size"])
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
        chunk_len = self.chunk_size - self.local_size
        inv_freq = self._build_inv_freq(query.device)
        q_pos = positions % chunk_len

        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        query_positions = (
            q_pos,
            torch.clamp(q_pos + chunk_len, max=self.chunk_size),
            torch.full_like(q_pos, self.chunk_size),
            q_pos + chunk_len,
            q_pos + self.chunk_size,
        )
        query_parts = []
        for cur_pos in query_positions:
            cos, sin = self._build_cos_sin(cur_pos, inv_freq)
            query_parts.append(self._apply_rotary_embedding(query_compute, cos, sin))

        key_pos = positions % chunk_len
        key_cos, key_sin = self._build_cos_sin(key_pos, inv_freq)
        key_out = self._apply_rotary_embedding(key_compute, key_cos, key_sin)
        query_out = torch.cat(query_parts, dim=-1)
        return query_out.to(self.output_dtype), key_out.to(self.output_dtype)

    def _build_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.rotary_dim
            )
        )

    def _build_cos_sin(
        self,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions_float = positions.to(device=inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions_float, inv_freq)
        return freqs.cos().to(self.compute_dtype), freqs.sin().to(self.compute_dtype)

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
