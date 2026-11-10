# transpose

## Function

Swaps two tensor dimensions, equivalent to `torch.transpose(...).contiguous()`.

## Inputs

- `tensor`: input tensor with shape `shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor whose dimensions `dim0` and `dim1` are swapped.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement two-axis tensor transpose in backend-native kernel code.
- Support positive and negative `dim0` and `dim1` values.
- The output should be contiguous, matching the reference behavior.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.transpose`, `Tensor.transpose`, `permute`, `contiguous` as the core computation, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
