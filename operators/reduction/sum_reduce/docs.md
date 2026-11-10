# sum_reduce

## Function

Computes the sum of tensor values along one or more dimensions, equivalent to `torch.sum`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor reduced along `dim`.
- If `keepdim` is true, reduced dimensions remain with size 1.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the reduction in backend-native kernel code.
- Support positive and negative `dim` values.
- Support multi-axis reduction when `dim` is a list.
- For FP16/BF16 inputs, keep reduction accumulators and block partial sums in FP32, then cast to the output dtype only when writing final outputs.
- Parallel reduction changes addition order; avoid accumulating or carrying partial sums in FP16/BF16, especially for large reduction dimensions.
- Do not call `torch.sum` or other PyTorch operators for the core computation in `ModelNew.forward`.
