# stack

## Function

Stacks multiple tensors along a new dimension, equivalent to `torch.stack`.

## Inputs

- `*tensors`: `num_tensors` input tensors, each with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor whose shape equals `input_shape` with a new dimension of size `num_tensors` inserted at `dim`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis tensor stacking in backend-native kernel code.
- Support positive and negative `dim` values.
- Valid `dim` values are in `[-rank - 1, rank]`, where `rank = len(input_shape)`.
- `num_tensors` is positive and known from the shape entry.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.stack`, `torch.cat`, `torch.concat`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
