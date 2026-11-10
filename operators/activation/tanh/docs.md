# tanh

## Function

Applies the hyperbolic tangent activation elementwise.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement tanh in backend-native kernel code as an elementwise activation.
- Do not call `torch.tanh` or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the tanh/exponential path in FP32 when practical, especially near the transition region before saturation.
