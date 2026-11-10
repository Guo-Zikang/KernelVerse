from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.kimi_delta_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Kimi-style delta attention with vector alpha decay and vector beta update gates."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.d_model = self.num_heads * self.head_dim
        self.feature_map = shape.get("feature_map", "elu")
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_alpha = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_beta = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        q = self._linear(tensor, self.w_q)
        k = self._linear(tensor, self.w_k)
        v = self._linear(tensor, self.w_v)
        alpha = torch.sigmoid(self._linear(tensor, self.w_alpha))
        beta = torch.sigmoid(self._linear(tensor, self.w_beta))

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        alpha = alpha.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1,
            2,
        )
        beta = beta.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1,
            2,
        )

        q = self._feature_map(q)
        k = self._feature_map(k)
        scale = self.head_dim ** -0.5
        q = (q * scale).float()
        k = (k * scale).float()
        v = v.float()
        alpha = alpha.float()
        beta = beta.float()

        state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim),
            dtype=torch.float32,
            device=tensor.device,
        )
        outputs = []

        for token_idx in range(seq_len):
            q_t = q[:, :, token_idx, :]
            k_t = k[:, :, token_idx, :]
            v_t = v[:, :, token_idx, :]
            alpha_t = alpha[:, :, token_idx, :]
            beta_t = beta[:, :, token_idx, :]

            predicted_v = torch.einsum("bhd,bhde->bhe", k_t, state)
            delta_t = v_t - predicted_v
            state = alpha_t.unsqueeze(-1) * state
            state = state + torch.einsum("bhd,bhe->bhde", k_t, beta_t * delta_t)
            output_t = torch.einsum("bhd,bhde->bhe", q_t, state)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = F.linear(output, self.w_o.weight.float(), None)
        return output.to(self.output_dtype)

    def _feature_map(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return F.elu(tensor) + 1
        if self.feature_map == "relu":
            return F.relu(tensor)
        return tensor

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        return F.linear(tensor, weight, None)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    d_model = num_heads * head_dim
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32).to(dtype)
    return (tensor,)
