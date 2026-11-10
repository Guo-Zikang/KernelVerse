# scatter_add

## Function

Adds source values into an input tensor along one dimension, equivalent to `torch.scatter_add`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `index`: int64 tensor with shape `input_shape`; each value selects the destination position along `dim`.
- `src`: source tensor with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `input_shape`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis indexed accumulation in backend-native kernel code.
- Support positive and negative `dim` values.
- Every index value is in `[0, input_shape[dim])`.
- Repeated destination indices are allowed and must accumulate all corresponding source values.
- For FP16/BF16 inputs, accumulate repeated-index contributions in FP32, then cast once to the output dtype.
- Preserve repeated-index semantics exactly; all contributions to the same destination must participate in the accumulation.
- Do not call `torch.scatter_add`, `torch.scatter`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
