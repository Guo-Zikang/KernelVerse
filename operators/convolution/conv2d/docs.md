# conv2d

## Function

Applies a 2D convolution to NCHW image-like feature maps, equivalent to `torch.nn.functional.conv2d`.

## Inputs

- `x`: input tensor with shape `[batch_size, in_channels, height, width]`.
- `weight`: convolution weight with shape `[out_channels, in_channels / groups, kernel_size, kernel_size]`.
- `bias`: bias with shape `[out_channels]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `stride`, `padding`, `dilation`, and `groups` are known from the shape entry.

## Output

- Tensor with shape `[batch_size, out_channels, output_height, output_width]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the convolution in backend-native kernel code for NCHW tensors.
- Do not call `torch.nn.functional.conv2d`, `torch.conv2d`, cuDNN, or other existing convolution libraries for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate input * weight products in FP32 or hardware mixed-precision accumulators.
- Add bias in FP32 after the convolution accumulation.
- Cast only the final output back to the configured dtype.
