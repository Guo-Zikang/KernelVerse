# masked_fill

## Function

Fills tensor elements selected by a boolean mask with a scalar value, equivalent to `tensor.masked_fill(mask, fill_value)`.

## Inputs

- `tensor`: input tensor with shape `shape`.
- `mask`: boolean tensor with shape `shape`.

All input tensors use the dtype selected by the shape entry except `mask`, which is boolean.

## Output

- Tensor with shape `shape`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement elementwise masked fill in backend-native kernel code.
- Broadcasting is not required for this operator package; `tensor` and `mask` have exactly the same shape.
- `fill_value` is a scalar numeric value from the shape entry.
- `mask_probability` controls the random true ratio used by `generate_inputs`.
- Preserve branch selection exactly; unmasked values must be copied without modification.
- For FP16/BF16 outputs, cast `fill_value` to the output dtype once and avoid extra intermediate rounding.
- Do not call `masked_fill`, `torch.where`, masking helpers, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
