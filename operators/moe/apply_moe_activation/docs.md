# apply_moe_activation

## Function

Applies MoE expert activation variants to token hidden states.

## Inputs

- `x`: input tensor.

For gated activation variants, `x` has shape `[num_tokens, 2 * hidden_dim]` and the last dimension stores gate and up branches.

For `_no_mul` activation variants, `x` has shape `[num_tokens, hidden_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns one tensor with shape `[num_tokens, hidden_dim]` and the configured shape dtype.

## Activation Variants

- `silu`: `silu(gate) * up`.
- `gelu`: `gelu(gate) * up`.
- `relu2`: `relu(gate)^2 * up`.
- `swigluoai`: clamped OpenAI-style SwiGLU variant.
- `swiglustep`: clamped stepped SwiGLU variant.
- `silu_no_mul`: SiLU without an up-branch multiply.
- `gelu_no_mul`: GELU without an up-branch multiply.
- `relu2_no_mul`: squared ReLU without an up-branch multiply.

## Implementation Notes

- Implement the selected activation in backend-native kernel code.
- Do not call PyTorch activation functions or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute sigmoid/SiLU/GELU/ReLU2 activation paths in FP32.
- Perform the gate-up multiplication in FP32 when practical.
- Cast only the final output back to the configured dtype.
