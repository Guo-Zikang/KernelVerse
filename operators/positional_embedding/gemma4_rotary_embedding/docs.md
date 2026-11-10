# gemma4_rotary_embedding

## Function

Applies Gemma4 rotary positional embedding to flattened query and key tensors.

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

- This is not exactly plain rotary embedding. Gemma4 uses `head_size` rather than `rotary_dim` as the frequency exponent denominator.
- Let `rope_angles = rotary_dim // 2` and `nope_angles = head_size // 2 - rope_angles`.
- Build inverse frequencies with `1 / base ** (arange(0, 2 * rope_angles, 2) / head_size)`.
- If `nope_angles > 0`, append zero inverse frequencies for the non-RoPE tail.
- Compute `freqs = positions outer inv_freq`, then `cos(freqs)` and `sin(freqs)`.
- Rotate the full `head_size` channels; zero inverse frequencies make the non-RoPE tail pass through unchanged.
- For Neox style, split the full head into two halves and rotate the two halves.
- For non-Neox style, rotate even and odd channels pairwise.
- Apply the same Gemma4 rotary transform independently to every query and key head.
- Do not replace this with plain rotary embedding unless `rotary_dim == head_size` and the frequency denominator is still handled correctly.
- Do not call PyTorch einsum, cos, sin, chunk, stack, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact frequency formula and Neox/non-Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
