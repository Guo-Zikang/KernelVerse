# index_select

## Function

Selects entries from an input tensor along one dimension, equivalent to `torch.index_select`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `index`: int64 1D tensor with shape `[index_size]`; each value selects an element along `dim`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor whose shape equals `input_shape` except dimension `dim` is replaced by `index_size`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis index selection in backend-native kernel code.
- Support positive and negative `dim` values.
- Every index value is in `[0, input_shape[dim])`.
- `index_size` is positive and may differ from `input_shape[dim]`; repeated source positions are allowed.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.index_select`, `torch.gather`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
