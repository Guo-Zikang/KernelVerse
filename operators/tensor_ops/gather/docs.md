# gather

## Function

Gathers values from an input tensor along one dimension, equivalent to `torch.gather`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `index`: int64 tensor with the same shape as `tensor`; each value selects an element along `dim`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as `index`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement indexed gather in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis gather is required for this operator.
- Every index value is in `[0, input_shape[dim])`.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.gather`, `torch.take_along_dim`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
