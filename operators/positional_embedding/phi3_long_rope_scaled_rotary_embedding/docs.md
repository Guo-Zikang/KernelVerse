# phi3_long_rope_scaled_rotary_embedding

## Function

Applies Phi-3 LongRoPE scaled rotary positional embedding to flattened query and key tensors.

## Inputs

- `positions`: int32 tensor with shape `[batch_size * seq_len]`.
- `query`: tensor with shape `[batch_size * seq_len, num_q_heads * head_size]`.
- `key`: tensor with shape `[batch_size * seq_len, num_kv_heads * head_size]`.

All floating input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `query_out`: tensor with the same shape as `query`.
- `key_out`: tensor with the same shape as `key`.

All returned tensors must use the configured dtype.

## Implementation Notes

- This is a Phi-3 LongRoPE variant, not plain RoPE, linear scaling RoPE, NTK scaling RoPE, or Llama 3 frequency scaling.
- Select `long_factor` when `use_long_rope` is true, otherwise select `short_factor`.
- Build inverse frequencies with `1 / (factor[i] * base ** (channel / rotary_dim))`.
- Apply `long_mscale` or `short_mscale` to the generated cos/sin tables after choosing the factor path.
- Only the first `rotary_dim` values of each head are rotated; the remaining `head_size - rotary_dim` values pass through unchanged.
- Use Neox-style rotation: split the rotary slice into two halves and rotate the two halves.
- Apply the same rotary transform independently to every query and key head.
- Do not call PyTorch einsum, cos, sin, chunk, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact short/long scaling factor selection and Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
