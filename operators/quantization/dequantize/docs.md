# dequantize

## Function

Converts a quantized tensor back to a floating-point tensor.

## Inputs

- `tensor`: quantized tensor with shape `shape`.
- The input may use `per_tensor` or `per_channel` affine quantization.

## Output

- Float tensor with the same shape as `tensor`.
- The output dtype is `FP32`.

## Implementation Notes

- Per-tensor dequantization uses one global `scale` and `zero_point`.
- Per-channel dequantization uses the channel dimension selected by `axis`.
- Dequantization semantics are equivalent to `(quantized_value - zero_point) * scale`.
- For `QINT8`, the stored integer range is `[-128, 127]`.
- For `QUINT8`, the stored integer range is `[0, 255]`.
- Do not call `tensor.dequantize`, PyTorch quantization helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Read the stored quantized integer value exactly before applying dequantization.
- Compute `(integer_value - zero_point) * scale` in FP32.
- For per-channel inputs, use the scale and zero_point selected by the normalized channel axis.
