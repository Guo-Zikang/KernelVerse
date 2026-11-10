# cat

## Function

Concatenates multiple tensors along one dimension, equivalent to `torch.cat`.

## Inputs

- `*tensors`: `num_tensors` input tensors, each with shape `input_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor whose shape equals `input_shape` except dimension `dim` is multiplied by `num_tensors`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis tensor concatenation in backend-native kernel code.
- Support positive and negative `dim` values.
- `num_tensors` is positive and known from the shape entry.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.cat`, `torch.concat`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
