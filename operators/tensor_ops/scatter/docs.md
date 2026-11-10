# scatter

## Function

Writes source values into an input tensor along one dimension, equivalent to `torch.scatter`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `index`: int64 tensor with shape `input_shape`; each value selects the destination position along `dim`.
- `src`: source tensor with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `input_shape`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis scatter assignment in backend-native kernel code.
- Support positive and negative `dim` values.
- Every index value is in `[0, input_shape[dim])`.
- The implementation should match `torch.scatter(tensor, dim, index, src)` semantics for this shape contract.
- Preserve assignment semantics exactly; destination elements selected by `index` receive `src` values without arithmetic or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values from `tensor` and `src`.
- Do not call `torch.scatter`, `torch.scatter_add`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
