from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.token_routing_sparse_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Token-level top-k routing attention with route-overlap sparse visibility."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.num_routes = int(shape["num_routes"])
        self.top_k_routes = int(shape["top_k_routes"])
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
        self.router = nn.Linear(self.d_model, self.num_routes, bias=False)

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

        route_indices = self._route_tokens(tensor)
        mask = self._make_route_overlap_mask(route_indices)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(
            ~mask.view(batch_size, 1, seq_len, seq_len),
            float("-inf"),
        )
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _route_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        router_weight = self.router.weight.to(self.compute_dtype)
        routing_logits = F.linear(tensor, router_weight, None)
        return torch.argsort(
            routing_logits,
            dim=-1,
            descending=True,
            stable=True,
        )[..., : self.top_k_routes]

    def _make_route_overlap_mask(self, route_indices: torch.Tensor) -> torch.Tensor:
        route_membership = F.one_hot(route_indices.long(), num_classes=self.num_routes).bool()
        route_membership = route_membership.any(dim=-2)
        mask = (route_membership.unsqueeze(2) & route_membership.unsqueeze(1)).any(dim=-1)
        seq_len = route_indices.size(1)
        diagonal = torch.eye(
            seq_len,
            dtype=torch.bool,
            device=route_indices.device,
        ).unsqueeze(0)
        return mask | diagonal

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
