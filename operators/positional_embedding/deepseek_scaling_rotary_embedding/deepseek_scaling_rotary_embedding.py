import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.deepseek_scaling_rotary_embedding.helpers import (
    DEFAULT_BASE,
    DEFAULT_IS_NEOX_STYLE,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply DeepSeek/YARN-scaled rotary embeddings to query and key tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.head_size = int(shape["head_size"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.max_position_embeddings = int(shape["max_position_embeddings"])
        self.base = float(shape.get("base", DEFAULT_BASE))
        self.scaling_factor = float(shape["scaling_factor"])
        self.extrapolation_factor = float(shape.get("extrapolation_factor", 1.0))
        self.attn_factor = float(shape.get("attn_factor", 1.0))
        self.beta_fast = int(shape.get("beta_fast", 32))
        self.beta_slow = int(shape.get("beta_slow", 1))
        self.is_neox_style = bool(shape.get("is_neox_style", DEFAULT_IS_NEOX_STYLE))
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        mscale = float(shape.get("mscale", 1.0))
        mscale_all_dim = float(shape.get("mscale_all_dim", 0.0))
        self.scale_multiplier = (
            self._deepseek_mscale(self.scaling_factor, mscale)
            / self._deepseek_mscale(self.scaling_factor, mscale_all_dim)
            * self.attn_factor
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_compute = query.to(self.compute_dtype)
        key_compute = key.to(self.compute_dtype)
        inv_freq = self._yarn_inv_freq(query.device)
        cos, sin = self._build_cos_sin(positions + offsets, inv_freq)
        cos = cos * self.scale_multiplier
        sin = sin * self.scale_multiplier

        query_out = self._apply_rotary_embedding(query_compute, cos, sin)
        key_out = self._apply_rotary_embedding(key_compute, cos, sin)
        return query_out.to(self.output_dtype), key_out.to(self.output_dtype)

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

    def _yarn_inv_freq(self, device: torch.device) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)
        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = (
            1.0 - self._yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, device)
        ) * self.extrapolation_factor
        return inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: int,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> float:
        return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    @classmethod
    def _yarn_find_correction_range(
        cls,
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> Tuple[int, int]:
        low = math.floor(cls._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(cls._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_linear_ramp_mask(low: int, high: int, dim: int, device: torch.device) -> torch.Tensor:
        if low == high:
            high += 1
        ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
        return torch.clamp(ramp, 0, 1)

    @staticmethod
    def _deepseek_mscale(scale: float, mscale: float) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    offsets = torch.randint(0, 4, positions.shape, dtype=torch.int32)
    return positions, query, key, offsets
