# split

## Function

Splits a tensor into chunks along one dimension and materializes each split as an output tensor.

## Inputs

- `tensor`: input tensor with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tuple of tensors split along `dim`.
- Every output tensor has the same shape as `input_shape` except dimension `dim` is at most `split_size`.
- The final output tensor may be smaller than `split_size` when `input_shape[dim]` is not divisible by `split_size`.
- Output dtype must match the configured shape dtype.
- Outputs must be materialized contiguous tensors, not view-only aliases of the input.

## Implementation Notes

- Implement single-axis tensor splitting in backend-native kernel code.
- Support positive and negative `dim` values.
- Copy the selected input ranges into independent contiguous output tensors.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.split`, `torch.chunk`, slicing helpers, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
