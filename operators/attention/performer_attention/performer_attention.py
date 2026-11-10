from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.attention.performer_attention.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Performer attention using positive random features for softmax kernel approximation."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(shape["head_dim"])
        self.num_features = int(shape.get("num_features", self.head_dim))
        self.eps = float(shape.get("eps", 1e-6))
        self.d_model = self.num_heads * self.head_dim
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

        projection_matrix = self._make_projection_matrix(
            self.num_features,
            self.head_dim,
            int(shape.get("random_seed", 0)),
        )
        self.register_buffer("projection_matrix", projection_matrix, persistent=False)

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

        projection = self.projection_matrix.to(self.compute_dtype)
        q_prime = self._softmax_kernel(q, projection, is_query=True)
        k_prime = self._softmax_kernel(k, projection, is_query=False)

        kv = torch.matmul(k_prime.transpose(-2, -1), v)
        normalizer = torch.einsum(
            "bhlm,bhm->bhl",
            q_prime,
            k_prime.sum(dim=2),
        ).unsqueeze(-1)
        output = torch.matmul(q_prime, kv) / normalizer.clamp(min=self.eps)

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        output = self._linear(output, self.w_o)
        return output.to(self.output_dtype)

    def _softmax_kernel(
        self,
        tensor: torch.Tensor,
        projection: torch.Tensor,
        is_query: bool,
    ) -> torch.Tensor:
        data = tensor * (self.head_dim ** -0.25)
        projected = torch.matmul(data, projection.transpose(-2, -1))
        squared_norm = (data * data).sum(dim=-1, keepdim=True) * 0.5
        features = projected - squared_norm
        if is_query:
            features = features - features.max(dim=-1, keepdim=True).values.detach()
        else:
            features = features - features.amax(dim=(-2, -1), keepdim=True).detach()
        return torch.exp(features) * (self.num_features ** -0.5) + self.eps

    def _make_projection_matrix(
        self,
        num_features: int,
        head_dim: int,
        random_seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        blocks = []
        remaining = num_features
        while remaining > 0:
            gaussian = torch.randn((head_dim, head_dim), generator=generator)
            q, _ = torch.linalg.qr(gaussian)
            block = q.transpose(0, 1)
            blocks.append(block[: min(remaining, head_dim)])
            remaining -= head_dim
        return torch.cat(blocks, dim=0)

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
