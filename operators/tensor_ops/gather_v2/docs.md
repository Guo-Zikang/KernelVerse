# gather_v2

## Function

Gathers slices from an embedding tensor along a selected axis, equivalent to GatherV2-style lookup.

## Inputs

- `params`: input tensor with shape `params_shape`.
- `indices`: integer tensor with shape `indices_shape`.

All floating input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Flatten `indices`, gather slices along `axis`, then reshape to the GatherV2 output shape.
- Support positive and negative `axis` values.
- Every index value is in `[0, params_shape[axis])`.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.index_select`, `torch.gather`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
