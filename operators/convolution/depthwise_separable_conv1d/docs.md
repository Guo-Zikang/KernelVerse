# depthwise_separable_conv1d

## Function

Applies a depthwise 1D convolution followed by a pointwise 1D convolution to token-major sequence features.

## Inputs

- `x`: input tensor with shape `[batch_size, seq_len, input_dim]`.
- `dw_weight`: depthwise convolution weight with shape `[input_dim * depthwise_multiplier, 1, kernel_size]`.
- `dw_bias`: depthwise convolution bias with shape `[input_dim * depthwise_multiplier]`.
- `pw_weight`: pointwise convolution weight with shape `[out_channels, input_dim * depthwise_multiplier, 1]`.
- `pw_bias`: pointwise convolution bias with shape `[out_channels]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `padding` is known from the shape entry.

## Output

- Tensor with shape `[batch_size, output_seq_len, out_channels]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the depthwise stage and pointwise stage in backend-native kernel code.
- Do not call `torch.nn.functional.conv1d`, `torch.permute`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate both the depthwise convolution and pointwise convolution in FP32 or hardware mixed-precision accumulators.
- Add depthwise and pointwise biases in FP32.
- Cast only the final output back to the configured dtype.
