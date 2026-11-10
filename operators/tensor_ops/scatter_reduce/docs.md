# scatter_reduce

## Function

Reduces source values into an input tensor along one dimension, equivalent to `Tensor.scatter_reduce`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `index`: int64 tensor with shape `input_shape`; each value selects the destination position along `dim`.
- `src`: source tensor with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `input_shape`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis scatter-reduce in backend-native kernel code.
- Support positive and negative `dim` values.
- Supported reductions are `sum`, `prod`, `mean`, `amax`, and `amin`.
- Respect `include_self`: when true, the original `tensor` value participates in the reduction; when false, only scattered `src` values participate for written positions.
- Every index value is in `[0, input_shape[dim])`.
- Repeated destination indices are allowed and must be reduced according to the configured `reduce` mode.
- For FP16/BF16 inputs, accumulate repeated-index contributions in FP32 for `sum`, `mean`, and additive paths, then cast once to the output dtype.
- For `amax` and `amin`, preserve comparison semantics exactly; for `prod`, avoid unnecessary FP16/BF16 intermediate rounding when practical.
- Do not call `torch.scatter_reduce`, `Tensor.scatter_reduce`, `torch.scatter`, `torch.scatter_add`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
