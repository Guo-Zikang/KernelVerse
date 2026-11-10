# causal_conv1d_update

## Function

Updates causal depthwise 1D convolution outputs using cached convolution state.

## Inputs

- `x`: new input tokens with shape `[batch, dim, seqlen]`.
- `conv_state`: cached previous tokens with shape `[batch, dim, state_len]`.
- `weight`: depthwise convolution weight with shape `[dim, width]`.
- `bias`: bias with shape `[dim]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `width`, `state_len`, `use_bias`, and `silu_activation` are known from the shape entry.
- `state_len` is at least `width - 1`.
- Each output timestep uses cached history plus the current update chunk, without reading future timesteps.

## Output

- Tensor with shape `[batch, dim, seqlen]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the cached causal depthwise convolution update in backend-native kernel code.
- If `silu_activation` is true, apply SiLU after the convolution accumulation.
- Do not call `torch.cat`, `torch.nn.functional.conv1d`, `torch.nn.functional.silu`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate the cached causal depthwise convolution over `width` in FP32.
- Add bias in FP32 before optional SiLU.
- Preserve `conv_state` cached values exactly in the configured dtype; use FP32 only for the output accumulation path.
- If `silu_activation` is true, compute SiLU in FP32 and cast only the final output back to the configured dtype.
