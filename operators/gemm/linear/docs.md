# linear

## Function

Computes a dense linear projection: `tensor @ weight.T + bias`.

## Inputs

- `tensor`: input tensor with shape `[..., in_features]`.
- `weight`: tensor with shape `[out_features, in_features]`.
- `bias`: tensor with shape `[out_features]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[..., out_features]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the projection and bias add in backend-native kernel code.
- Do not call `torch.nn.functional.linear`, `torch.matmul`, `torch.mm`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- For FP16/BF16 inputs, accumulate the K dimension in FP32 or hardware mixed-precision matrix accumulators, then add bias and cast once to the output dtype.
- Use the `[out_features, in_features]` weight layout directly without materializing a transposed copy.
- Use tiled kernels with coalesced memory access and shared-memory reuse; use hardware matrix acceleration for aligned FP16/BF16 shapes when practical, with correct fallback for tail and unaligned sizes.
