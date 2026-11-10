# matmul_v2

## Function

Computes dense matrix multiplication where the second input is stored in transposed projection-weight layout: `tensor_a @ tensor_b.T`.

## Inputs

- `tensor_a`: tensor with shape `[M, K]`.
- `tensor_b`: tensor with shape `[N, K]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[M, N]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- This operator differs from `matmul`: the second input is stored as `[N, K]`, not `[K, N]`.
- Implement the matrix multiply in backend-native kernel code using the transposed layout directly.
- Do not call `torch.matmul`, `torch.mm`, transpose kernels, cuBLAS, CUTLASS, or other operator libraries for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- For FP16/BF16 inputs, accumulate the K dimension in FP32 or hardware mixed-precision matrix accumulators, then cast once to the output dtype.
- Use the `[N, K]` second-input layout directly and avoid materializing a transposed copy.
- Use tiled kernels with coalesced memory access and shared-memory reuse; use hardware matrix acceleration for aligned FP16/BF16 shapes when practical, with correct fallback for tail and unaligned sizes.
