# llama3_rotary_embedding

## Function

Applies Llama 3 scaled rotary positional embedding to flattened query and key tensors.

## Inputs

- `positions`: int32 tensor with shape `[batch_size * seq_len]`.
- `query`: tensor with shape `[batch_size * seq_len, num_q_heads * head_size]`.
- `key`: tensor with shape `[batch_size * seq_len, num_kv_heads * head_size]`.

Floating input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `query_out`: tensor with the same shape as `query`.
- `key_out`: tensor with the same shape as `key`.

All returned tensors must use the configured dtype.

## Implementation Notes

- This is not plain rotary embedding. It applies Llama 3 frequency-dependent scaling to inverse frequencies.
- Build the original inverse frequencies with `1 / base ** (arange(0, rotary_dim, 2) / rotary_dim)`.
- Compute `wave_len = 2 * pi / inv_freq`.
- Frequencies with `wave_len < original_max_position_embeddings / high_freq_factor` remain unchanged.
- Frequencies with `wave_len > original_max_position_embeddings / low_freq_factor` are divided by `scaling_factor`.
- Intermediate frequencies use smooth interpolation between the scaled and unscaled inverse frequencies.
- Compute `freqs = positions outer scaled_inv_freq`, then `cos(freqs)` and `sin(freqs)`.
- Only the first `rotary_dim` values of each head are rotated; the remaining `head_size - rotary_dim` values pass through unchanged.
- For Neox style, split the rotary slice into two halves and rotate the two halves.
- For non-Neox style, rotate even and odd channels pairwise.
- Do not call PyTorch einsum, cos, sin, chunk, stack, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact Llama 3 frequency scaling formula and Neox/non-Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
