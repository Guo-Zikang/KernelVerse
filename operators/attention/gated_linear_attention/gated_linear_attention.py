from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.gated_linear_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Recurrent gated linear attention with vector gate state decay."""

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
        self.w_g = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        q = self._linear(tensor, self.w_q)
        k = self._linear(tensor, self.w_k)
        v = self._linear(tensor, self.w_v)
        gate = torch.sigmoid(self._linear(tensor, self.w_g))

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        gate = gate.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1,
            2,
        )

        q = self._feature_map(q)
        k = self._feature_map(k)

        state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim),
            dtype=self.compute_dtype,
            device=tensor.device,
        )
        outputs = []

        for token_idx in range(seq_len):
            q_t = q[:, :, token_idx, :]
            k_t = k[:, :, token_idx, :]
            v_t = v[:, :, token_idx, :]
            gate_t = gate[:, :, token_idx, :]

            kv_t = torch.einsum("bhd,bhe->bhde", k_t, v_t)
            state = gate_t.unsqueeze(-1) * state + kv_t
            output_t = torch.einsum("bhd,bhde->bhe", q_t, state)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
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
