# per_token_group_quant_int8

## Function

Quantizes each token row into signed INT8 values with one FP32 scale per hidden group.

## Inputs

- `input_tensor`: tensor with shape `[num_tokens, hidden_dim]`.

The hidden dimension is split into contiguous groups of `group_size`.

## Output

Returns a tuple:

- `q`: INT8 tensor with shape `[num_tokens, hidden_dim]`.
- `scales`: FP32 tensor with shape `[num_tokens, hidden_dim / group_size]`.

## Implementation Notes

- For each token and group, compute `scale = max(abs(group)) / 127`.
- Clamp `scale` to at least `eps`.
- Compute `q = round(group / scale).clamp(-127, 127).to(int8)`.
- Implement group max, scale computation, quantization, and writeback in backend-native kernel code.
- Do not call PyTorch amax, round, clamp, quantization helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For each token group, compute `max(abs(group))` and `scale` in FP32.
- Divide group values by the FP32 scale before rounding.
- Clamp rounded values to `[-127, 127]` before writing INT8, and store scales as FP32.
