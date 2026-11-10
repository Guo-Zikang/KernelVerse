from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.nystromformer_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Nyströmformer attention using landmark pooling and iterative pseudo-inverse."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.num_landmarks = int(shape["num_landmarks"])
        self.pinv_iterations = int(shape.get("pinv_iterations", 6))
        self.eps = float(shape.get("eps", 1e-6))
        self.d_model = self.num_heads * self.head_dim
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
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

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

        if seq_len <= self.num_landmarks:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
        else:
            q_landmarks = self._pool_landmarks(q)
            k_landmarks = self._pool_landmarks(k)

            kernel_1 = F.softmax(
                torch.matmul(q, k_landmarks.transpose(-2, -1))
                * (self.head_dim ** -0.5),
                dim=-1,
            )
            kernel_2 = F.softmax(
                torch.matmul(q_landmarks, k_landmarks.transpose(-2, -1))
                * (self.head_dim ** -0.5),
                dim=-1,
            )
            kernel_3 = F.softmax(
                torch.matmul(q_landmarks, k.transpose(-2, -1))
                * (self.head_dim ** -0.5),
                dim=-1,
            )

            kernel_2_inv = self._iterative_pinv(kernel_2)
            output = torch.matmul(
                torch.matmul(kernel_1, kernel_2_inv),
                torch.matmul(kernel_3, v),
            )

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _pool_landmarks(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        segment_size = (seq_len + self.num_landmarks - 1) // self.num_landmarks
        padded_len = segment_size * self.num_landmarks
        if padded_len > seq_len:
            pad_len = padded_len - seq_len
            tensor = F.pad(tensor, (0, 0, 0, pad_len))
        tensor = tensor.view(
            batch_size,
            num_heads,
            self.num_landmarks,
            segment_size,
            head_dim,
        )
        return tensor.mean(dim=3)

    def _iterative_pinv(self, matrix: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
        identity = identity.expand(*matrix.shape[:-2], matrix.size(-1), matrix.size(-1))
        matrix = matrix + self.eps * identity
        col_norm = matrix.abs().sum(dim=-2).amax(dim=-1, keepdim=True)
        row_norm = matrix.abs().sum(dim=-1).amax(dim=-1, keepdim=True)
        z = matrix.transpose(-2, -1) / (
            col_norm * row_norm
        ).clamp(min=self.eps).unsqueeze(-1)

        for _ in range(self.pinv_iterations):
            matrix_z = torch.matmul(matrix, z)
            z = 0.25 * torch.matmul(
                z,
                13 * identity
                - torch.matmul(
                    matrix_z,
                    15 * identity - torch.matmul(matrix_z, 7 * identity - matrix_z),
                ),
            )
        return z

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
