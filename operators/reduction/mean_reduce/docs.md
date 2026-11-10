# mean_reduce

## Function

Computes the mean of tensor values along one or more dimensions, equivalent to `torch.mean`.

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
- Divide by the number of reduced elements to match mean semantics.
- For FP16/BF16 inputs, keep reduction accumulators and block partial sums in FP32, then divide in FP32 before casting to the output dtype.
- Parallel reduction changes addition order; avoid accumulating or carrying partial sums in FP16/BF16, especially for large reduction dimensions.
- Do not call `torch.mean` or other PyTorch operators for the core computation in `ModelNew.forward`.
