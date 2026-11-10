# permute

## Function

Reorders tensor dimensions using a full permutation, equivalent to `Tensor.permute(...).contiguous()`.

## Inputs

- `tensor`: input tensor with shape `shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor whose dimensions are reordered by `perm`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement full-rank dimension permutation in backend-native kernel code.
- `perm` is a complete permutation of `[0, rank)`.
- The output should be contiguous, matching the reference behavior.
- This is a data movement operator: copy selected input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values; do not upcast and recompute values.
- Do not call `torch.permute`, `Tensor.permute`, `transpose`, `contiguous` as the core computation, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
