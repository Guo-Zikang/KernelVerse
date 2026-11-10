from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.logits_sampling.masked_softmax.helpers import (
    _normalize_dim,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Apply softmax over allowed positions selected by a boolean mask."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.dim = _normalize_dim(shape["dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor_compute = tensor.to(self.compute_dtype)
        mask_bool = mask.bool()
        fill_value = torch.finfo(self.compute_dtype).min
        masked_tensor = tensor_compute.masked_fill(~mask_bool, fill_value)
        output = F.softmax(masked_tensor, dim=self.dim)
        output = output.masked_fill(~mask_bool, 0.0)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    tensor_shape = tuple(int(dim) for dim in shape["shape"])
    dim = _normalize_dim(shape["dim"])
    if dim < 0:
        dim += len(tensor_shape)
    dtype = _resolve_dtype(shape["dtype"])
    keep_probability = float(shape.get("keep_probability", 0.75))
    tensor = torch.randn(tensor_shape, dtype=torch.float32).to(dtype)
    mask = torch.rand(tensor_shape) < keep_probability
    mask.select(dim, 0).fill_(True)
    return tensor, mask
