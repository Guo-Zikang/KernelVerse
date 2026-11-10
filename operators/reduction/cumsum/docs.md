# cumsum

## Function

Computes the cumulative sum of tensor values along one dimension, equivalent to `torch.cumsum`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the prefix-sum scan in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis scan is required for this operator.
- For low-precision inputs such as FP16 or BF16, keep the running prefix sum and any block carry values in FP32, then cast each final prefix output back to the required dtype.
- Parallel scan changes the addition order; keep the scan numerically close to `torch.cumsum` for long axes, especially for BF16, and avoid accumulating or carrying partial sums in FP16/BF16.
- Do not call `torch.cumsum`, `torch.sum`, or other PyTorch operators for the core computation in `ModelNew.forward`.
