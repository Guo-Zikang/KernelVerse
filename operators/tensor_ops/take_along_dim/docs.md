# take_along_dim

## Function

Takes values from an input tensor along one dimension, equivalent to `torch.take_along_dim`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `indices`: int64 tensor with the same shape as `tensor`; each value selects an element along `dim`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as `indices`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement indexed take-along-dim in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis take-along-dim is required for this operator.
- Every index value is in `[0, input_shape[dim])`.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.take_along_dim`, `torch.gather`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
