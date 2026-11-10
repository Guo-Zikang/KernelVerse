# sigmoid

## Function

Applies the sigmoid activation elementwise.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement sigmoid in backend-native kernel code as an elementwise activation.
- Do not call `torch.sigmoid` or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the exponential/sigmoid math in FP32 when practical, then cast the final output to the configured dtype.
