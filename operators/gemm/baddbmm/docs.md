# baddbmm

## Function

Computes `beta * input_tensor + alpha * (batch1 @ batch2)` for each batch.

## Inputs

- `input_tensor`: tensor with shape `[batch_size, M, N]`.
- `batch1`: tensor with shape `[batch_size, M, K]`.
- `batch2`: tensor with shape `[batch_size, K, N]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, M, N]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the batched matrix multiply, scaling, and add in backend-native kernel code.
- Do not call `torch.baddbmm`, `torch.bmm`, `torch.matmul`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- For FP16/BF16 inputs, accumulate `batch1 @ batch2` in FP32 or hardware mixed-precision matrix accumulators, then apply `alpha`, `beta`, and the input add before casting once to the output dtype.
- Preserve `alpha` and `beta` semantics exactly for every batch, including `alpha=0`, `beta=0`, negative values, and non-1 scaling.
- Use tiled kernels with coalesced memory access and shared-memory reuse; use hardware matrix acceleration for aligned FP16/BF16 shapes when practical, with correct fallback for tail and unaligned sizes.
