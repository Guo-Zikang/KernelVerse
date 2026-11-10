from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.synthesizer_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Dense Synthesizer attention with MLP-generated attention logits."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.seq_len = int(shape["seq_len"])
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.hidden_dim = int(shape["hidden_dim"])
        self.d_model = self.num_heads * self.head_dim
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.synth_fc1 = nn.Linear(self.d_model, self.hidden_dim, bias=True)
        self.synth_fc2 = nn.Linear(
            self.hidden_dim,
            self.num_heads * self.seq_len,
            bias=True,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        value = self._linear(tensor, self.w_v)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        hidden = F.relu(self._linear(tensor, self.synth_fc1))
        logits = self._linear(hidden, self.synth_fc2)
        logits = logits.view(batch_size, seq_len, self.num_heads, self.seq_len)
        logits = logits.permute(0, 2, 1, 3)

        attn_weights = F.softmax(logits, dim=-1)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        bias = layer.bias.to(self.compute_dtype) if layer.bias is not None else None
        return F.linear(tensor, weight, bias)


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
