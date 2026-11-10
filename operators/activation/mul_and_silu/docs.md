# mul_and_silu

## Function

Applies a fused gated activation. The input is split into two equal halves along the last dimension, then computes `value * silu(gate)`.

## Inputs

- `x`: input tensor with shape from the selected benchmark shape.

The last dimension is split as `[value, gate]`.

## Output

- Tensor with the same rank as the input.
- The output size along the last dimension is half of the input size.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the split, SiLU activation, and multiply in backend-native kernel code.
- The first half of the last dimension is `value`; the second half is `gate`.
- Compute `value * gate * sigmoid(gate)`.
- Do not call `torch.nn.functional.silu`, `torch.sigmoid`, slicing helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the SiLU gate and `value * silu(gate)` multiply in FP32 when practical, then cast the final output.
