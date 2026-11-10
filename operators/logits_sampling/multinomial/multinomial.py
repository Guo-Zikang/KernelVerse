from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.multinomial.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Deterministic multinomial sampling driven by explicit uniform random values."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_samples = int(shape["num_samples"])
        self.replacement = bool(shape.get("replacement", False))

    def forward(self, weights: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        weights_compute = weights.to(torch.float32)
        uniforms_compute = uniforms.to(torch.float32).clamp(
            min=0.0,
            max=1.0 - torch.finfo(torch.float32).eps,
        )

        if self.replacement:
            return self._sample_with_replacement(weights_compute, uniforms_compute)
        return self._sample_without_replacement(weights_compute, uniforms_compute)

    @staticmethod
    def _sample_from_cdf(weights: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        cdf = torch.cumsum(weights, dim=-1)
        total = cdf[..., -1]
        target = uniforms * total
        squeeze_result = weights.ndim > 1 and target.ndim == weights.ndim - 1
        if squeeze_result:
            target = target.unsqueeze(-1)
        sampled = torch.searchsorted(cdf.contiguous(), target.contiguous(), right=False)
        if squeeze_result:
            sampled = sampled.squeeze(-1)
        return sampled.clamp(max=weights.size(-1) - 1).to(torch.int64)

    def _sample_with_replacement(
        self,
        weights: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> torch.Tensor:
        if weights.ndim == 1:
            return self._sample_from_cdf(weights, uniforms)
        cdf = torch.cumsum(weights, dim=-1)
        total = cdf[..., -1:].expand_as(uniforms)
        target = uniforms * total
        sampled = torch.searchsorted(cdf.contiguous(), target.contiguous(), right=False)
        return sampled.clamp(max=weights.size(-1) - 1).to(torch.int64)

    def _sample_without_replacement(
        self,
        weights: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> torch.Tensor:
        work_weights = weights.clone()
        outputs = []
        if weights.ndim == 1:
            for sample_idx in range(self.num_samples):
                sampled = self._sample_from_cdf(work_weights, uniforms[sample_idx])
                outputs.append(sampled)
                work_weights[sampled] = 0.0
            return torch.stack(outputs).to(torch.int64)

        batch_indices = torch.arange(weights.size(0), device=weights.device)
        for sample_idx in range(self.num_samples):
            sampled = self._sample_from_cdf(work_weights, uniforms[:, sample_idx])
            outputs.append(sampled)
            work_weights[batch_indices, sampled] = 0.0
        return torch.stack(outputs, dim=1).to(torch.int64)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    num_samples = int(shape["num_samples"])
    dtype = _resolve_dtype(shape["dtype"])
    weights = torch.rand(tensor_shape, dtype=torch.float32) + 1e-6
    uniform_shape = (num_samples,) if len(tensor_shape) == 1 else (tensor_shape[0], num_samples)
    uniforms = torch.rand(uniform_shape, dtype=torch.float32)
    return weights.to(dtype), uniforms
