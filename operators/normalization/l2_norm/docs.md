# l2_norm

## Function

Computes L2 normalization over the last tensor dimension.

## Inputs

- `tensor`: input tensor with shape `shape`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `eps` is known from the shape entry.

## Output

- Tensor with the same shape as `tensor`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Normalize over the last dimension only.
- Compute `norm = sqrt(sum(tensor ** 2, dim=-1, keepdim=True))`.
- Clamp `norm` to at least `eps`, then compute `tensor / norm`.
- Do not call PyTorch normalization, norm, sum, pow, sqrt, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute the squared sum over the last dimension in FP32.
- Apply `sqrt`, `clamp_min(eps)`, and division in FP32.
- Cast only the final output back to the configured dtype.
