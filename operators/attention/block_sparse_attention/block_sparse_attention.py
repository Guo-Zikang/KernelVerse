from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.block_sparse_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Block sparse attention with fixed, random, top-k, or local-plus-top-k block selection."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.block_size = int(shape["block_size"])
        self.num_selected_blocks = int(shape["num_selected_blocks"])
        self.selection_mode = str(shape["selection_mode"])
        self.num_blocks = int(shape["seq_len"]) // self.block_size
        self.d_model = self.num_heads * self.head_dim
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        if self.selection_mode in {"fixed", "random"}:
            indices = self._make_static_block_indices(int(shape.get("random_seed", 0)))
            self.register_buffer("static_block_indices", indices, persistent=False)
        else:
            self.static_block_indices = None

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

        q_blocks = q.view(
            batch_size,
            self.num_heads,
            self.num_blocks,
            self.block_size,
            self.head_dim,
        )
        k_blocks = k.view(
            batch_size,
            self.num_heads,
            self.num_blocks,
            self.block_size,
            self.head_dim,
        )
        v_blocks = v.view(
            batch_size,
            self.num_heads,
            self.num_blocks,
            self.block_size,
            self.head_dim,
        )
        block_indices = self._select_blocks(q_blocks, k_blocks)
        output_blocks = torch.empty_like(q_blocks)

        for batch_idx in range(batch_size):
            for head_idx in range(self.num_heads):
                for query_block_idx in range(self.num_blocks):
                    selected_blocks = block_indices[
                        batch_idx, head_idx, query_block_idx
                    ].long()
                    selected_key = k_blocks[batch_idx, head_idx, selected_blocks]
                    selected_value = v_blocks[batch_idx, head_idx, selected_blocks]
                    selected_key = selected_key.reshape(-1, self.head_dim)
                    selected_value = selected_value.reshape(-1, self.head_dim)
                    query_block = q_blocks[batch_idx, head_idx, query_block_idx]
                    scores = torch.matmul(query_block, selected_key.transpose(-2, -1))
                    scores = scores * (self.head_dim ** -0.5)
                    attn_weights = F.softmax(scores, dim=-1)
                    output_blocks[batch_idx, head_idx, query_block_idx] = torch.matmul(
                        attn_weights,
                        selected_value,
                    )

        output = output_blocks.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _select_blocks(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = q_blocks.size(0)
        if self.selection_mode in {"fixed", "random"}:
            return self.static_block_indices.to(q_blocks.device).view(
                1, 1, self.num_blocks, self.num_selected_blocks
            ).expand(batch_size, self.num_heads, -1, -1)

        q_summary = q_blocks.mean(dim=3)
        k_summary = k_blocks.mean(dim=3)
        block_scores = torch.matmul(q_summary, k_summary.transpose(-2, -1))

        if self.selection_mode == "topk":
            return torch.argsort(
                block_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[..., : self.num_selected_blocks]

        local_indices = torch.arange(self.num_blocks, device=q_blocks.device).view(
            1, 1, self.num_blocks, 1
        ).expand(batch_size, self.num_heads, -1, -1)
        if self.num_selected_blocks == 1:
            return local_indices

        scores = block_scores.clone()
        scores.scatter_(-1, local_indices, torch.finfo(scores.dtype).min)
        topk_indices = torch.argsort(
            scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., : self.num_selected_blocks - 1]
        return torch.cat((local_indices, topk_indices), dim=-1)

    def _make_static_block_indices(self, random_seed: int) -> torch.Tensor:
        if self.selection_mode == "fixed":
            query_blocks = torch.arange(self.num_blocks, dtype=torch.long).view(
                self.num_blocks,
                1,
            )
            offsets = torch.arange(self.num_selected_blocks, dtype=torch.long).view(
                1,
                self.num_selected_blocks,
            )
            return (query_blocks + offsets) % self.num_blocks

        generator = torch.Generator()
        generator.manual_seed(int(random_seed))
        rows = []
        for _ in range(self.num_blocks):
            rows.append(torch.randperm(self.num_blocks, generator=generator)[
                : self.num_selected_blocks
            ])
        return torch.stack(rows, dim=0)

    def _linear(self, tensor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        weight = layer.weight.to(self.compute_dtype)
        return F.linear(tensor, weight, None)


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
