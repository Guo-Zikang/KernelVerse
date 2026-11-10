# where

## Function

Selects values elementwise from two tensors according to a boolean condition, equivalent to `torch.where(condition, tensor, other)`.

## Inputs

- `condition`: boolean tensor with shape `shape`.
- `tensor`: input tensor with shape `shape`.
- `other`: input tensor with shape `shape`.

All input tensors use the dtype selected by the shape entry except `condition`, which is boolean.

## Output

- Tensor with shape `shape`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement elementwise conditional selection in backend-native kernel code.
- `condition_probability` controls the random true ratio used by `generate_inputs`.
- Preserve branch selection exactly; each output element must come from the selected input branch without modifying unselected values.
- For FP16/BF16 inputs, preserve the original stored values from `tensor` and `other`.
- Do not call `torch.where`, masking helpers, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
