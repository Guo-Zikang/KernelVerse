# quantize_per_channel

## Function

Applies per-channel affine quantization to a floating-point tensor.

## Inputs

- `tensor`: float tensor with shape `shape`.

## Output

- Plain integer storage tensor with the same shape as `tensor`, without PyTorch quantized-tensor metadata.
- `QINT8` produces `torch.int8`; `QUINT8` produces `torch.uint8`.

## Implementation Notes

- Normalize negative `axis` values to the corresponding positive axis.
- The number of per-channel quantization parameters is `tensor.size(axis)`.
- The scalar `scale` in the benchmark shape is a base scale. Channel `c` uses `scale * (1 + (c % 8) / 8)` so channels have genuinely different scales.
- The scalar `zero_point` is expanded to a per-channel vector of length `tensor.size(axis)`.
- Quantization semantics are equivalent to `round(tensor / scale) + zero_point`, clamped to the selected quantized dtype range for each channel.
- `QINT8` uses the integer range `[-128, 127]`.
- `QUINT8` uses the integer range `[0, 255]`.
- Do not call `torch.quantize_per_channel`, PyTorch quantization helpers, clamp, round, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert input values to FP32 before per-channel scale, zero_point, rounding, and clamping.
- Use the channel-specific scale and zero_point selected by the normalized channel axis for each element.
- Match PyTorch affine quantization rounding behavior; do not use truncation or floor.
