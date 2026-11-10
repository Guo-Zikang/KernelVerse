# quantize_per_tensor

## Function

Applies per-tensor affine quantization to a floating-point tensor.

## Inputs

- `tensor`: float tensor with shape `shape`.

## Output

- Quantized tensor with the same shape as `tensor`.
- Output dtype is selected by `quantized_dtype`, either `QINT8` or `QUINT8`.

## Implementation Notes

- Use one global `scale` and `zero_point` for the entire tensor.
- Quantization semantics are equivalent to `round(tensor / scale) + zero_point`, clamped to the selected quantized dtype range.
- `QINT8` uses the integer range `[-128, 127]`.
- `QUINT8` uses the integer range `[0, 255]`.
- Do not call `torch.quantize_per_tensor`, PyTorch quantization helpers, clamp, round, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert input values to FP32 before applying scale, zero_point, rounding, and clamping.
- Match PyTorch affine quantization rounding behavior; do not use truncation or floor.
- Clamp after adding zero_point, and clamp to the exact output dtype range.
