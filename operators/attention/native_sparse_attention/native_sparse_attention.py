from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.native_sparse_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Native sparse attention reference with compressed, selected, and local branches."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.seq_len = int(shape["seq_len"])
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.block_size = int(shape["block_size"])
        self.num_selected_blocks = int(shape["num_selected_blocks"])
        self.local_window_size = int(shape["local_window_size"])
        self.num_blocks = self.seq_len // self.block_size
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
        tensor_compute = tensor.to(self.compute_dtype)
        batch_size = tensor_compute.size(0)
        seq_len = tensor_compute.size(1)

        query = self._linear(tensor_compute, self.w_q)
        key = self._linear(tensor_compute, self.w_k)
        value = self._linear(tensor_compute, self.w_v)

        query = query.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_blocks = key.view(
            batch_size,
            self.num_heads,
            self.num_blocks,
            self.block_size,
            self.head_dim,
        )
        value_blocks = value.view(
            batch_size,
            self.num_heads,
            self.num_blocks,
            self.block_size,
            self.head_dim,
        )
        compressed_key = key_blocks.mean(dim=3)
        compressed_value = value_blocks.mean(dim=3)

        compressed_output = self._compressed_attention(query, compressed_key, compressed_value)
        selected_output = self._selected_attention(
            query,
            key_blocks,
            value_blocks,
            compressed_key,
        )
        local_output = self._local_attention(query, key, value)
        output = compressed_output + selected_output + local_output

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _compressed_attention(
        self,
        query: torch.Tensor,
        compressed_key: torch.Tensor,
        compressed_value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = query.shape
        output = torch.zeros_like(query)
        scale = self.head_dim ** -0.5

        for batch_idx in range(batch_size):
            for head_idx in range(self.num_heads):
                for query_idx in range(seq_len):
                    valid_blocks = self._completed_blocks(query_idx, query.device)
                    if valid_blocks.numel() == 0:
                        continue
                    query_token = query[batch_idx, head_idx, query_idx]
                    selected_key = compressed_key[batch_idx, head_idx, valid_blocks]
                    selected_value = compressed_value[batch_idx, head_idx, valid_blocks]
                    scores = torch.matmul(query_token, selected_key.transpose(-2, -1))
                    scores = scores * scale
                    attn_weights = F.softmax(scores, dim=-1)
                    output[batch_idx, head_idx, query_idx] = torch.matmul(
                        attn_weights,
                        selected_value,
                    )

        return output

    def _selected_attention(
        self,
        query: torch.Tensor,
        key_blocks: torch.Tensor,
        value_blocks: torch.Tensor,
        compressed_key: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = query.shape
        output = torch.zeros_like(query)
        scale = self.head_dim ** -0.5

        for batch_idx in range(batch_size):
            for head_idx in range(self.num_heads):
                for query_idx in range(seq_len):
                    valid_blocks = self._completed_blocks(query_idx, query.device)
                    if valid_blocks.numel() == 0:
                        continue
                    query_token = query[batch_idx, head_idx, query_idx]
                    block_keys = compressed_key[batch_idx, head_idx, valid_blocks]
                    block_scores = torch.matmul(query_token, block_keys.transpose(-2, -1))
                    k_blocks = min(self.num_selected_blocks, valid_blocks.numel())
                    selected_offsets = torch.argsort(
                        block_scores,
                        dim=-1,
                        descending=True,
                        stable=True,
                    )[:k_blocks]
                    selected_blocks = valid_blocks[selected_offsets]
                    selected_key = key_blocks[
                        batch_idx,
                        head_idx,
                        selected_blocks,
                    ].reshape(-1, self.head_dim)
                    selected_value = value_blocks[
                        batch_idx,
                        head_idx,
                        selected_blocks,
                    ].reshape(-1, self.head_dim)
                    scores = torch.matmul(query_token, selected_key.transpose(-2, -1))
                    scores = scores * scale
                    attn_weights = F.softmax(scores, dim=-1)
                    output[batch_idx, head_idx, query_idx] = torch.matmul(
                        attn_weights,
                        selected_value,
                    )

        return output

    def _local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = query.shape
        output = torch.empty_like(query)
        scale = self.head_dim ** -0.5

        for query_idx in range(seq_len):
            key_start = max(0, query_idx - self.local_window_size + 1)
            key_end = query_idx + 1
            query_token = query[:, :, query_idx : query_idx + 1, :]
            local_key = key[:, :, key_start:key_end, :]
            local_value = value[:, :, key_start:key_end, :]
            scores = torch.matmul(query_token, local_key.transpose(-2, -1))
            scores = scores * scale
            attn_weights = F.softmax(scores, dim=-1)
            output[:, :, query_idx : query_idx + 1, :] = torch.matmul(
                attn_weights,
                local_value,
            )

        return output

    def _completed_blocks(self, query_idx: int, device: torch.device) -> torch.Tensor:
        block_ids = torch.arange(self.num_blocks, device=device)
        block_ends = (block_ids + 1) * self.block_size
        return block_ids[block_ends <= query_idx + 1]

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
