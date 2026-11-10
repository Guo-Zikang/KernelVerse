# glu_pointwise_conv

## Function

Applies a 1D convolution followed by a gated linear unit to token-major sequence features.

## Inputs

- `x`: input tensor with shape `[batch_size, seq_len, input_dim]`.
- `conv_weight`: convolution weight with shape `[2 * output_dim, input_dim, kernel_size]`.
- `conv_bias`: convolution bias with shape `[2 * output_dim]`.
- `glu_bias_left`: optional GLU value-branch bias with shape `[1, output_dim, 1]`.
- `glu_bias_right`: optional GLU gate-branch bias with shape `[1, output_dim, 1]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `bias_in_glu` and `causal` are known from the shape entry.
- If `causal` is true, the convolution uses left causal padding and trims future positions.

## Output

- Tensor with shape `[batch_size, output_seq_len, output_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the convolution, branch split, sigmoid gate, and elementwise product in backend-native kernel code.
- Do not call `torch.nn.functional.conv1d`, `torch.sigmoid`, `torch.permute`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate the convolution in FP32 or hardware mixed-precision accumulators.
- Add convolution bias and optional GLU branch biases in FP32.
- Compute the sigmoid gate and gate-value multiplication in FP32, then cast only the final output back to the configured dtype.
