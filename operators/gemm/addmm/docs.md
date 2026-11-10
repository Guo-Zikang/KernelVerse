# addmm

## Function

Computes `beta * input + alpha * (mat1 @ mat2)`.

## Inputs

- `input_tensor`: tensor with shape `[M, N]`.
- `mat1`: tensor with shape `[M, K]`.
- `mat2`: tensor with shape `[K, N]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[M, N]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the matrix multiply and add in backend-native kernel code.
- Do not call `torch.addmm`, `torch.matmul`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- For FP16/BF16 inputs, accumulate `mat1 @ mat2` in FP32 or hardware mixed-precision matrix accumulators, then apply `alpha`, `beta`, and the input add before casting once to the output dtype.
- Preserve `alpha` and `beta` semantics exactly, including `alpha=0`, `beta=0`, negative values, and non-1 scaling.
- Use tiled kernels with coalesced memory access and shared-memory reuse; use hardware matrix acceleration for aligned FP16/BF16 shapes when practical, with correct fallback for tail and unaligned sizes.
