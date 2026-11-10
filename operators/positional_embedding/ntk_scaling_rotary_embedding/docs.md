# ntk_scaling_rotary_embedding

## Function

Applies NTK-scaled rotary positional embedding to flattened query and key tensors.

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

- This is NTK-scaled RoPE with optional mixed NTK scaling.
- Without `mixed_b`, use `scaled_base = base * scaling_factor`, build inverse frequencies from `scaled_base`, then divide by `scaling_factor ** (2 / rotary_dim)`.
- With `mixed_b`, use the original base and divide each frequency by `lambda_1_m = exp(log(scaling_factor) / (rotary_dim / 2) ** mixed_b * arange(1, rotary_dim // 2 + 1) ** mixed_b)`.
- Compute `freqs = positions outer ntk_scaled_inv_freq`, then `cos(freqs)` and `sin(freqs)`.
- Only the first `rotary_dim` values of each head are rotated; the remaining `head_size - rotary_dim` values pass through unchanged.
- For Neox style, split the rotary slice into two halves and rotate the two halves.
- For non-Neox style, rotate even and odd channels pairwise.
- Do not call PyTorch einsum, cos, sin, chunk, stack, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact NTK scaling formula and Neox/non-Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
