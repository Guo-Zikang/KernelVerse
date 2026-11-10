from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.moe.apply_moe_activation.helpers import (
    GLU_ACTIVATIONS,
    NO_MUL_ACTIVATIONS,
    _resolve_dtype,
    activation_input_width,
    validate_shape,
)


class Model(nn.Module):
    """Applies the selected MoE activation to input tensors."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.activation = shape["activation"]
        self.output_dtype = _resolve_dtype(shape["dtype"])
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.activation
        x_compute = x.to(self.compute_dtype)

        if activation in GLU_ACTIVATIONS:
            gate, up = torch.chunk(x_compute, 2, dim=-1)
            if activation == "silu":
                out = F.silu(gate) * up
            elif activation == "gelu":
                out = F.gelu(gate, approximate="none") * up
            elif activation == "relu2":
                out = torch.square(F.relu(gate)) * up
            elif activation == "swigluoai":
                gate_clamped = gate.clamp(max=7.0)
                up_clamped = up.clamp(min=-7.0, max=7.0)
                out = (up_clamped + 1.0) * (gate_clamped * torch.sigmoid(gate_clamped * 1.702))
            else:
                gate_silu = gate * torch.sigmoid(gate)
                gate_clamped = gate_silu.clamp(max=7.0)
                up_clamped = up.clamp(min=-7.0, max=7.0)
                out = gate_clamped * up_clamped
            return out.to(self.output_dtype)

        if activation in NO_MUL_ACTIVATIONS:
            if activation == "silu_no_mul":
                out = F.silu(x_compute)
            elif activation == "gelu_no_mul":
                out = F.gelu(x_compute, approximate="none")
            else:
                out = torch.square(F.relu(x_compute))
            return out.to(self.output_dtype)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    num_tokens = int(shape["num_tokens"])
    hidden_dim = int(shape["hidden_dim"])
    activation = shape["activation"]
    width = activation_input_width(hidden_dim, activation)
    x = torch.randn((num_tokens, width), dtype=torch.float32).to(dtype)
    return (x,)
