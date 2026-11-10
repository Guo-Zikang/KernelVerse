from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.tensor_ops.gather_v2.helpers import (
    _normalize_axis,
    _resolve_float_dtype,
    _resolve_index_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Gather slices from an embedding tensor along a selected axis."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.axis = _normalize_axis(int(shape["axis"]), len(shape["params_shape"]))
        self.output_dtype = _resolve_float_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        params_compute = params.to(self.compute_dtype)
        flat_indices = indices.reshape(-1).long()
        gathered = torch.index_select(params_compute, dim=self.axis, index=flat_indices)
        output_shape = (
            list(params.shape[: self.axis])
            + list(indices.shape)
            + list(params.shape[self.axis + 1 :])
        )
        return gathered.reshape(output_shape).to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    params_shape = tuple(int(dim) for dim in shape["params_shape"])
    indices_shape = tuple(int(dim) for dim in shape["indices_shape"])
    axis = _normalize_axis(int(shape["axis"]), len(params_shape))
    dtype = _resolve_float_dtype(shape["dtype"])
    indices_dtype = _resolve_index_dtype(shape["indices_dtype"])

    params = torch.randn(
        params_shape,
        dtype=torch.float32,
    ).to(dtype)
    indices = torch.randint(
        0,
        params_shape[axis],
        indices_shape,
        dtype=indices_dtype,
    )
    return params, indices
