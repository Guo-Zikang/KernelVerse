# static_scaled_int8_quant

## Function

Quantizes a floating-point tensor to ordinary INT8 values using a static scale and zero point.

## Inputs

- `x`: input tensor with shape from the selected benchmark shape.
- `scale`: FP32 tensor with one element.
- `azp`: INT32 tensor with one element.

The input tensor uses the dtype selected by the shape entry.

## Output

- `q`: INT8 tensor with the same shape as `x`.

## Implementation Notes

- Compute `round(x / max(scale, MIN_SCALE) + azp)`.
- Clamp quantized values to `[-127, 127]`.
- Output ordinary `torch.int8` values, not a PyTorch quantized tensor.
- Implement the scale, zero-point addition, rounding, clamping, and INT8 write in backend-native kernel code.
- Do not call PyTorch quantization helpers, round, clamp, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert `x`, `scale`, and `azp` to FP32 for `x / max(scale, MIN_SCALE) + azp`.
- Apply rounding before the INT8 cast; do not truncate.
- Clamp in FP32 to `[-127, 127]`, then write INT8.
