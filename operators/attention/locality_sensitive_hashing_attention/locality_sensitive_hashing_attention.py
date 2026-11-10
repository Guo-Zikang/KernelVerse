from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.locality_sensitive_hashing_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Locality-sensitive hashing attention semantic reference."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.num_hashes = int(shape["num_hashes"])
        self.num_buckets = int(shape["num_buckets"])
        self.hash_seed = int(shape.get("hash_seed", 0))
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
        generator.manual_seed(self.hash_seed)
        random_rotations = torch.randn(
            self.num_hashes,
            self.num_heads,
            self.head_dim,
            self.num_buckets,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer("random_rotations", random_rotations)

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

        mask = self._create_lsh_mask(q)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _create_lsh_mask(self, query: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, _ = query.shape
        if self.num_buckets == 1:
            return torch.ones(
                (batch_size, num_heads, seq_len, seq_len),
                dtype=torch.bool,
                device=query.device,
            )

        rotations = self.random_rotations.to(device=query.device, dtype=self.compute_dtype)
        hash_scores = torch.einsum("bhld,hdr->bhlr", query, rotations[0])
        bucket_ids = torch.argsort(
            hash_scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., 0].unsqueeze(0)

        for hash_index in range(1, self.num_hashes):
            hash_scores = torch.einsum("bhld,hdr->bhlr", query, rotations[hash_index])
            next_bucket_ids = torch.argsort(
                hash_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[..., 0].unsqueeze(0)
            bucket_ids = torch.cat((bucket_ids, next_bucket_ids), dim=0)

        mask = torch.zeros(
            (batch_size, num_heads, seq_len, seq_len),
            dtype=torch.bool,
            device=query.device,
        )
        for hash_index in range(self.num_hashes):
            ids = bucket_ids[hash_index]
            mask = mask | (ids.unsqueeze(-1) == ids.unsqueeze(-2))

        token_indices = torch.arange(seq_len, device=query.device)
        mask[:, :, token_indices, token_indices] = True
        return mask

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
