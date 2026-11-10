# dynamic_ntk_scaling_rotary_embedding

## Function

Applies dynamic NTK-scaled rotary positional embedding to flattened query and key tensors.

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

- This is not plain rotary embedding. It recomputes RoPE frequencies using a dynamically enlarged NTK base.
- Compute `max_len = max_position_embeddings * scaling_factor`.
- Compute `dynamic_base = base * (scaling_factor * max_len / max_position_embeddings - (scaling_factor - 1.0)) ** (rotary_dim / (rotary_dim - 2.0))`.
- Build inverse frequencies with `1 / dynamic_base ** (arange(0, rotary_dim, 2) / rotary_dim)`.
- Compute `freqs = positions outer inv_freq`, then `cos(freqs)` and `sin(freqs)`.
- Only the first `rotary_dim` values of each head are rotated; the remaining `head_size - rotary_dim` values pass through unchanged.
- For Neox style, split the rotary slice into two halves and rotate the two halves.
- For non-Neox style, rotate even and odd channels pairwise.
- Apply the same dynamic NTK-scaled rotary transform independently to every query and key head.
- Do not replace this with plain rotary embedding unless `dynamic_base` mathematically equals the configured base.
- Do not call PyTorch einsum, cos, sin, chunk, stack, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact dynamic NTK scaling formula and Neox/non-Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
