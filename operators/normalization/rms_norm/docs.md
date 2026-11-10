# rms_norm

## Function

Computes RMS normalization over the last tensor dimension.

## Inputs

- `tensor`: input tensor with shape `shape`.
- `weight`: scale tensor with shape `[shape[-1]]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as `tensor`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Normalize over the last dimension only.
- Compute `variance = mean(tensor ** 2, dim=-1, keepdim=True)`.
- Compute `normalized = tensor * rsqrt(variance + eps)`.
- Multiply the normalized tensor by `weight`.
- Do not call PyTorch RMSNorm, mean, pow, rsqrt, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute `mean(tensor * tensor)` over the last dimension in FP32.
- Apply `eps`, `rsqrt`, normalization, and weight scaling in FP32.
- Cast only the final output back to the configured dtype.
