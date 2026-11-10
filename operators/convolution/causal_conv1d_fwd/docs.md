# causal_conv1d_fwd

## Function

Applies a causal depthwise 1D convolution over sequence positions.

## Inputs

- `x`: input tensor with shape `[batch, dim, seqlen]`.
- `weight`: depthwise convolution weight with shape `[dim, width]`.
- `bias`: bias with shape `[dim]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `width`, `use_bias`, and `silu_activation` are known from the shape entry.
- Each output timestep only uses the current and previous input timesteps.

## Output

- Tensor with shape `[batch, dim, seqlen]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the causal depthwise convolution in backend-native kernel code.
- If `silu_activation` is true, apply SiLU after the convolution accumulation.
- Do not call `torch.nn.functional.pad`, `torch.nn.functional.conv1d`, `torch.nn.functional.silu`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate the causal depthwise convolution over `width` in FP32.
- Add bias in FP32 before optional SiLU.
- If `silu_activation` is true, compute SiLU in FP32 and cast only the final output back to the configured dtype.
