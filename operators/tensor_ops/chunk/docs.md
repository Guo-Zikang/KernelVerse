# chunk

## Function

Splits a tensor into a requested number of chunks along one dimension and materializes each chunk as an output tensor.

## Inputs

- `tensor`: input tensor with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tuple of tensors chunked along `dim`.
- Output tensors differ from `input_shape` only on dimension `dim`.
- The number of returned tensors may be smaller than `chunks` when PyTorch `torch.chunk` semantics allow it.
- Output dtype must match the configured shape dtype.
- Outputs must be materialized contiguous tensors, not view-only aliases of the input.

## Implementation Notes

- Implement single-axis tensor chunking in backend-native kernel code.
- Support positive and negative `dim` values.
- Copy the selected input ranges into independent contiguous output tensors.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.chunk`, `torch.split`, slicing helpers, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
