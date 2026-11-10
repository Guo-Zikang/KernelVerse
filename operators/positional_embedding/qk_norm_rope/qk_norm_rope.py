from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.positional_embedding.qk_norm_rope.helpers import _resolve_dtype, validate_shape


DEFAULT_BASE = 10000.0


class Model(nn.Module):
    """Fuses Q/K RMSNorm and rotary embedding over a packed QKV tensor."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads_q = int(shape["num_heads_q"])
        self.num_heads_k = int(shape["num_heads_k"])
        self.num_heads_v = int(shape["num_heads_v"])
        self.head_dim = int(shape["head_dim"])
        self.rotary_dim = int(shape["rotary_dim"])
        self.eps = float(shape["eps"])
        self.is_neox = bool(shape["is_neox"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q_size = self.num_heads_q * self.head_dim
        k_size = self.num_heads_k * self.head_dim
        q = qkv[:, :q_size]
        k = qkv[:, q_size : q_size + k_size]
        v = qkv[:, q_size + k_size :]

        q_view = q.reshape(-1, self.head_dim).to(self.compute_dtype)
        q_weight_compute = q_weight.to(self.compute_dtype)
        q_variance = q_view.pow(2).mean(dim=-1, keepdim=True)
        q_norm = (q_view * torch.rsqrt(q_variance + self.eps) * q_weight_compute).reshape_as(q)

        k_view = k.reshape(-1, self.head_dim).to(self.compute_dtype)
        k_weight_compute = k_weight.to(self.compute_dtype)
        k_variance = k_view.pow(2).mean(dim=-1, keepdim=True)
        k_norm = (k_view * torch.rsqrt(k_variance + self.eps) * k_weight_compute).reshape_as(k)

        positions = position_ids.to(device=qkv.device, dtype=torch.float32)
        inv_freq = 1.0 / DEFAULT_BASE ** (
            torch.arange(
                0,
                self.rotary_dim,
                2,
                device=qkv.device,
                dtype=torch.float32,
            )
            / self.rotary_dim
        )
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos().to(self.compute_dtype).unsqueeze(-2)
        sin = freqs.sin().to(self.compute_dtype).unsqueeze(-2)

        q_view = q_norm.reshape(q_norm.shape[0], -1, self.head_dim)
        k_view = k_norm.reshape(k_norm.shape[0], -1, self.head_dim)
        q_rot = q_view[..., : self.rotary_dim]
        k_rot = k_view[..., : self.rotary_dim]
        q_pass = q_view[..., self.rotary_dim :]
        k_pass = k_view[..., self.rotary_dim :]

        if self.is_neox:
            q1, q2 = torch.chunk(q_rot, 2, dim=-1)
            k1, k2 = torch.chunk(k_rot, 2, dim=-1)
            q_out = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
            k_out = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
        else:
            q1 = q_rot[..., ::2]
            q2 = q_rot[..., 1::2]
            k1 = k_rot[..., ::2]
            k2 = k_rot[..., 1::2]
            q_out = torch.stack((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1).flatten(-2)
            k_out = torch.stack((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1).flatten(-2)

        q_rope = torch.cat((q_out, q_pass), dim=-1).reshape_as(q_norm)
        k_rope = torch.cat((k_out, k_pass), dim=-1).reshape_as(k_norm)
        fused = torch.cat((q_rope, k_rope, v.to(self.compute_dtype)), dim=-1)
        return fused.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_tokens = int(shape["num_tokens"])
    num_heads_q = int(shape["num_heads_q"])
    num_heads_k = int(shape["num_heads_k"])
    num_heads_v = int(shape["num_heads_v"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    qkv = torch.randn((num_tokens, hidden_size), dtype=torch.float32).to(dtype)
    q_weight = (torch.randn((head_dim,), dtype=torch.float32) * 5.0).to(dtype)
    k_weight = (torch.randn((head_dim,), dtype=torch.float32) * 5.0).to(dtype)
    position_ids = torch.arange(num_tokens, dtype=torch.int32) + 100
    return qkv, q_weight, k_weight, position_ids
