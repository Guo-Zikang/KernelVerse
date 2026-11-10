# hadamard_transform

## Function

Applies a Fast Hadamard Transform over the last tensor dimension. The last dimension must be a power of two.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as `tensor`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the butterfly add/subtract stages over the last dimension in backend-native kernel code.
- If `normalize` is true, divide the final result by `sqrt(shape[-1])`.
- For FP16/BF16 inputs, keep all butterfly stage values in FP32 across every stage, then cast to the output dtype only after the optional normalization.
- Do not store intermediate butterfly values in FP16/BF16 shared memory between stages; repeated low-precision stage rounding causes large mismatch rates.
- Do not call PyTorch transform, reshape-heavy helper kernels, or other PyTorch operators for the core computation in `ModelNew.forward`.
