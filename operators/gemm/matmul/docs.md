# matmul

## Function

Computes dense matrix multiplication: `tensor_a @ tensor_b`.

## Inputs

- `tensor_a`: tensor with shape `[M, K]`.
- `tensor_b`: tensor with shape `[K, N]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[M, N]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the matrix multiply in backend-native kernel code.
- Do not call `torch.matmul`, `torch.mm`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- For FP16/BF16 inputs, accumulate the K dimension in FP32 or hardware mixed-precision matrix accumulators, then cast once to the output dtype.
- Use tiled kernels with coalesced memory access and shared-memory reuse; use hardware matrix acceleration for aligned FP16/BF16 shapes when practical, with correct fallback for tail and unaligned sizes.
