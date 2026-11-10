from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.big_bird_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """BigBird-style attention with local, global, and random sparse links."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.window_size = int(shape["window_size"])
        self.num_random_blocks = int(shape["num_random_blocks"])
        self.seq_len = int(shape["seq_len"])
        self.random_seed = int(shape.get("random_seed", 0))
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

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)
        random_index_table = torch.stack(
            [
                torch.randperm(self.seq_len, generator=generator, dtype=torch.long)
                for _ in range(self.seq_len)
            ]
        )
        self.register_buffer("random_index_table", random_index_table)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        q = self._linear(tensor, self.w_q)
        k = self._linear(tensor, self.w_k)
        v = self._linear(tensor, self.w_v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = self._create_bigbird_mask(seq_len, tensor.device).unsqueeze(0).unsqueeze(0)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _create_bigbird_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        half_window = self.window_size // 2

        for index in range(seq_len):
            start = max(0, index - half_window)
            end = min(seq_len, index + half_window + 1)
            mask[index, start:end] = True

        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        random_index_table = self.random_index_table.to(device=device)
        for index in range(1, seq_len - 1):
            mask[index, random_index_table[index, :self.num_random_blocks]] = True

        return mask

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        bias = layer.bias.to(self.compute_dtype) if layer.bias is not None else None
        return F.linear(tensor, weight, bias)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    num_heads = int(shape["num_heads"])
    head_dim = int(shape["head_dim"])
    d_model = num_heads * head_dim
    dtype = _resolve_dtype(shape["dtype"])
    tensor = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32).to(dtype)
    return (tensor,)
