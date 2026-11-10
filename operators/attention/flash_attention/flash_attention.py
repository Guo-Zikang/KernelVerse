from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.flash_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Flash attention with tiled online softmax and block-level trace outputs."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.d_model = self.num_heads * self.head_dim
        self.block_size_q = int(shape["block_size_q"])
        self.block_size_kv = int(shape["block_size_kv"])
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

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = tensor.to(self.compute_dtype)
        batch_size = tensor.size(0)
        seq_len = tensor.size(1)

        q = self._linear(tensor, self.w_q)
        k = self._linear(tensor, self.w_k)
        v = self._linear(tensor, self.w_v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        output, block_maxes, block_exp_sums, block_output_sums = self._flash_attention(
            q,
            k,
            v,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return (
            output.to(self.output_dtype),
            block_maxes,
            block_exp_sums,
            block_output_sums,
        )

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len_q, _ = query.shape
        seq_len_kv = key.size(2)
        block_q = min(self.block_size_q, seq_len_q)
        block_kv = min(self.block_size_kv, seq_len_kv)
        num_q_blocks = (seq_len_q + block_q - 1) // block_q
        num_kv_blocks = (seq_len_kv + block_kv - 1) // block_kv

        output = torch.zeros_like(query)
        block_maxes = torch.full(
            (batch_size, num_heads, num_q_blocks, num_kv_blocks, block_q),
            -float("inf"),
            dtype=self.compute_dtype,
            device=query.device,
        )
        block_exp_sums = torch.zeros(
            (batch_size, num_heads, num_q_blocks, num_kv_blocks, block_q),
            dtype=self.compute_dtype,
            device=query.device,
        )
        block_output_sums = torch.zeros(
            (batch_size, num_heads, num_q_blocks, num_kv_blocks, block_q),
            dtype=self.compute_dtype,
            device=query.device,
        )

        for q_block_idx, q_start in enumerate(range(0, seq_len_q, block_q)):
            q_end = min(q_start + block_q, seq_len_q)
            query_block = query[:, :, q_start:q_end, :]
            q_block_len = q_end - q_start

            output_block = torch.zeros_like(query_block)
            normalizer = torch.zeros(
                (batch_size, num_heads, q_block_len),
                dtype=self.compute_dtype,
                device=query.device,
            )
            max_score = torch.full(
                (batch_size, num_heads, q_block_len),
                -float("inf"),
                dtype=self.compute_dtype,
                device=query.device,
            )

            for kv_block_idx, kv_start in enumerate(range(0, seq_len_kv, block_kv)):
                kv_end = min(kv_start + block_kv, seq_len_kv)
                key_block = key[:, :, kv_start:kv_end, :]
                value_block = value[:, :, kv_start:kv_end, :]

                scores = torch.matmul(query_block, key_block.transpose(-2, -1))
                scores = scores * (self.head_dim ** -0.5)
                block_max = scores.max(dim=-1).values
                block_exp = torch.exp(scores - block_max.unsqueeze(-1))
                block_weighted = torch.matmul(block_exp, value_block)
                block_maxes[:, :, q_block_idx, kv_block_idx, :q_block_len] = (
                    block_max
                )
                block_exp_sums[:, :, q_block_idx, kv_block_idx, :q_block_len] = (
                    block_exp.sum(dim=-1)
                )
                block_output_sums[:, :, q_block_idx, kv_block_idx, :q_block_len] = (
                    block_weighted.sum(dim=-1)
                )

                next_max_score = torch.maximum(max_score, scores.max(dim=-1).values)
                correction = torch.exp(max_score - next_max_score)
                probabilities = torch.exp(scores - next_max_score.unsqueeze(-1))
                next_normalizer = correction * normalizer + probabilities.sum(dim=-1)
                output_block = (
                    correction.unsqueeze(-1) * output_block
                    + torch.matmul(probabilities, value_block)
                )
                max_score = next_max_score
                normalizer = next_normalizer

            output[:, :, q_start:q_end, :] = output_block / normalizer.clamp(
                min=1e-6
            ).unsqueeze(-1)

        return (
            output,
            block_maxes.to(self.output_dtype),
            block_exp_sums.to(self.output_dtype),
            block_output_sums.to(self.output_dtype),
        )

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
