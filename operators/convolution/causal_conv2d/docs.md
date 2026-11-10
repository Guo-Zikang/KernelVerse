# causal_conv2d

## Function

Applies a width-causal 2D convolution to NCHW feature maps.

## Inputs

- `x`: input tensor with shape `[batch_size, in_channels, height, width]`.
- `weight`: convolution weight with shape `[out_channels, in_channels, kernel_size, kernel_size]`.
- `bias`: bias with shape `[out_channels]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `kernel_size` and `stride` are known from the shape entry.
- The operator pads only the width axis with `left_padding = kernel_size - 1` and `right_padding = stride - 1` before convolution.

## Output

- Tensor with shape `[batch_size, out_channels, output_height, output_width]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the width-causal padding and convolution in backend-native kernel code for NCHW tensors.
- Do not call `torch.nn.functional.pad`, `torch.nn.functional.conv2d`, `torch.conv2d`, cuDNN, or other existing convolution libraries for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate input * weight products in FP32 or hardware mixed-precision accumulators.
- Padding values must be exact zero and must not contribute to the accumulation.
- Add bias in FP32 after the convolution accumulation, then cast only the final output back to the configured dtype.
