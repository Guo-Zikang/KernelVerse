from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.tensor_ops.embedding.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Embedding lookup using torch.nn.functional.embedding."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight_compute = weight.to(self.compute_dtype)
        output = F.embedding(input_ids.long(), weight_compute)
        return output.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    input_shape = tuple(int(dim) for dim in shape["input_shape"])
    num_embeddings = int(shape["num_embeddings"])
    embedding_dim = int(shape["embedding_dim"])
    dtype = _resolve_dtype(shape["dtype"])
    input_ids = torch.randint(0, num_embeddings, input_shape, dtype=torch.long)
    weight = torch.randn((num_embeddings, embedding_dim), dtype=torch.float32).to(dtype)
    return input_ids, weight
