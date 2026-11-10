from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.etc_global_local_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """ETC-style global-local sparse attention semantic reference."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.window_size = int(shape["window_size"])
        self.d_model = self.num_heads * self.head_dim
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(
        self,
        regular_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        regular_tokens = regular_tokens.to(self.compute_dtype)
        global_tokens = global_tokens.to(self.compute_dtype)
        batch_size = regular_tokens.size(0)
        seq_len = regular_tokens.size(1)
        num_global_tokens = global_tokens.size(1)
        combined_len = seq_len + num_global_tokens
        combined = torch.cat((regular_tokens, global_tokens), dim=1)

        q = self._linear(combined, self.w_q)
        k = self._linear(combined, self.w_k)
        v = self._linear(combined, self.w_v)

        q = q.view(batch_size, combined_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, combined_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, combined_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = self._create_etc_mask(seq_len, num_global_tokens, combined.device)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0) == 0,
            torch.finfo(scores.dtype).min,
        )
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, combined_len, self.d_model)
        output = self._linear(output, self.w_o).to(self.output_dtype)
        updated_regular = output[:, :seq_len, :]
        updated_global = output[:, seq_len:, :]
        return updated_regular, updated_global

    def _create_etc_mask(
        self,
        seq_len: int,
        num_global_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        combined_len = seq_len + num_global_tokens
        mask = torch.zeros((combined_len, combined_len), dtype=torch.bool, device=device)
        half_window = self.window_size // 2

        for index in range(seq_len):
            start = max(0, index - half_window)
            end = min(seq_len, index + half_window + 1)
            mask[index, start:end] = True

        mask[:seq_len, seq_len:] = True
        mask[seq_len:, :seq_len] = True
        mask[seq_len:, seq_len:] = True
        return mask

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        bias = layer.bias.to(self.compute_dtype) if layer.bias is not None else None
        return F.linear(tensor, weight, bias)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_global_tokens = int(shape["num_global_tokens"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    d_model = num_heads * head_dim
    dtype = _resolve_dtype(shape["dtype"])
    regular_tokens = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32).to(dtype)
    global_tokens = torch.randn(
        (batch_size, num_global_tokens, d_model),
        dtype=torch.float32,
    ).to(dtype)
    return regular_tokens, global_tokens
