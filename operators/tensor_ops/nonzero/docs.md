# nonzero

## Function

Returns the coordinates of non-zero elements in row-major order.

## Inputs

- `x`: input tensor with shape `input_dims`.

The selected shape controls only the number of non-zero elements through `target_nonzero_count`. Non-zero positions are randomly distributed by `generate_inputs`.

## Output

Returns a coordinate tensor with shape `[num_nonzero, rank]` and dtype `index_dtype`.

## Implementation Notes

- Match `torch.nonzero(x, as_tuple=False)` ordering.
- Output rows must be sorted in row-major input traversal order.
- `target_nonzero_count` is used only to control input sparsity and output row count.
- Do not synthesize coordinates from `target_nonzero_count`; the implementation must read `x` and detect which elements are non-zero.
- Assuming the first `target_nonzero_count` flattened elements are non-zero is incorrect.
- Detect nonzero elements by reading actual input values with native `x != 0` semantics; NaN values should follow PyTorch nonzero behavior.
- Coordinate outputs must be exact INT64 row-major coordinates; do not approximate or infer them from shape metadata.
- Implement non-zero detection, prefix/count, and coordinate writeback in backend-native kernel code.
- Do not call PyTorch `nonzero`, compact helpers, prefix-scan helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.
