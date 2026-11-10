# dynamic_scaled_int8_quant

## Function

Dynamically computes one global scale for the entire input tensor and quantizes the input to signed INT8 values.

This is symmetric dynamic INT8 quantization. The returned activation zero point is always zero.

## Inputs

- `x`: input tensor with shape `dims`.

`x` uses the dtype selected by the shape entry.

## Output

Returns a tuple:

- `q`: INT8 tensor with shape `dims`.
- `scale`: FP32 scale tensor with shape `[1]`.
- `azp`: INT32 activation zero-point tensor with shape `[1]`.

## Implementation Notes

- Compute one global dynamic scale from the maximum absolute value of the whole tensor.
- Clamp scale by `MIN_SCALING_FACTOR`.
- Quantize by clipping to the signed INT8 range `[-127, 127]` and rounding to nearest integer.
- Return `azp` as zero.
- Do not call PyTorch amax, clamp, round, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute the global `amax(abs(x))` reduction in FP32, especially for FP16/BF16 inputs.
- Compute `scale = max(amax / 127, MIN_SCALING_FACTOR)` in FP32.
- Quantize from FP32 values with rounding and clamp to `[-127, 127]` before writing INT8.
