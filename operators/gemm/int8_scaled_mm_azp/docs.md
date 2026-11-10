# int8_scaled_mm_azp

## Function

Computes scaled INT8 matrix multiplication with optional activation zero-point correction and optional output bias.

## Inputs

- `a`: INT8 activation matrix with shape `[M, K]`.
- `b`: INT8 weight matrix with shape `[K, N]`.
- `a_scales`: activation scale tensor with shape `[M]`.
- `b_scales`: weight scale tensor with shape `[N]`.
- `azp_adj`: precomputed activation zero-point adjustment with shape `[N]`.
- `azp`: activation zero-point tensor with shape `[N]`.
- `bias`: output bias tensor with shape `[N]`.

## Output

Returns one tensor with shape `[M, N]`.

The output uses `output_dtype`.

## Implementation Notes

- Implement INT8 scaled GEMM, activation zero-point correction, zero-point adjustment, and optional bias in backend-native kernel code.
- If `has_azp` is false, ignore `azp` and do not center `b`.
- If `has_bias` is false, ignore `bias`.
- Do not call cuBLAS, CUTLASS, vendor GEMM libraries, `torch.matmul`, `torch.mm`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- Accumulate INT8 products in INT32, then apply scales, activation zero-point correction, optional adjustment, and optional bias in FP32 before casting to the output dtype.
- Preserve `has_azp` and `has_bias` semantics exactly; ignored inputs must not affect the result.
- Use tiled kernels with coalesced INT8 loads and shared-memory reuse; handle tail and unaligned M/N/K sizes correctly.
