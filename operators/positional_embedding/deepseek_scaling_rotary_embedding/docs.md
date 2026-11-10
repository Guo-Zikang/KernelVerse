# deepseek_scaling_rotary_embedding

## Function

Applies DeepSeek/YARN-scaled rotary positional embedding to flattened query and key tensors.

## Inputs

- `positions`: int32 tensor with shape `[batch_size * seq_len]`.
- `query`: tensor with shape `[batch_size * seq_len, num_q_heads * head_size]`.
- `key`: tensor with shape `[batch_size * seq_len, num_kv_heads * head_size]`.
- `offsets`: int32 tensor with shape `[batch_size * seq_len]`.

Floating input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `query_out`: tensor with the same shape as `query`.
- `key_out`: tensor with the same shape as `key`.

All returned tensors must use the configured dtype.

## Implementation Notes

- This is not plain RoPE. It uses YARN-style frequency interpolation and extrapolation for extended context.
- Build inverse frequencies from `base`, `scaling_factor`, `max_position_embeddings`, `beta_fast`, `beta_slow`, and `extrapolation_factor`.
- Use `positions + offsets` when computing the cosine and sine tables.
- Apply the DeepSeek scale multiplier to both cosine and sine values.
- Only the first `rotary_dim` values of each head are rotated; the remaining `head_size - rotary_dim` values pass through unchanged.
- For Neox style, split the rotary slice into two halves and rotate the two halves.
- For non-Neox style, rotate even and odd channels pairwise.
- Apply the same scaled rotary transform independently to every query and key head.
- Do not call PyTorch einsum, cos, sin, chunk, stack, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact scaling formula and Neox/non-Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
