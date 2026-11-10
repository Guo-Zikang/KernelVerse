# gelu

## Function

Applies the Gaussian Error Linear Unit activation elementwise.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement GELU in backend-native kernel code as an elementwise activation.
- Use the tanh approximation formula for GELU computation.
- Do not call `torch.nn.functional.gelu`, `torch.gelu`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the tanh-approx GELU polynomial and tanh argument in FP32 when practical; preserve the tanh approximation formula exactly.
