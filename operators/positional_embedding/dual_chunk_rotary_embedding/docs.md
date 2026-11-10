# dual_chunk_rotary_embedding

## Function

Builds dual-chunk rotary query variants and rotary-encoded keys for dual-chunk attention.

## Inputs

- `positions`: int32 tensor with shape `[batch_size * seq_len]`.
- `query`: tensor with shape `[batch_size * seq_len, num_q_heads * head_size]`.
- `key`: tensor with shape `[batch_size * seq_len, num_kv_heads * head_size]`.

Floating input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `query_out`: tensor with shape `[batch_size * seq_len, 5 * num_q_heads * head_size]`.
- `key_out`: tensor with the same shape as `key`.

All returned tensors must use the configured dtype.

## Implementation Notes

- This is not plain rotary embedding. Query output concatenates five rotary variants for different dual-chunk relations.
- Let `chunk_len = chunk_size - local_size` and `q_pos = positions % chunk_len`.
- Query variants use positions `q_pos`, `min(q_pos + chunk_len, chunk_size)`, `chunk_size`, `q_pos + chunk_len`, and `q_pos + chunk_size`.
- Keys use `positions % chunk_len`.
- Only the first `rotary_dim` values of each head are rotated; the remaining `head_size - rotary_dim` values pass through unchanged.
- For Neox style, split the rotary slice into two halves and rotate the two halves.
- For non-Neox style, rotate even and odd channels pairwise.
- Do not collapse the five query variants into one output; the concatenated query output shape is part of the operator contract.
- Do not call PyTorch einsum, cos, sin, chunk, stack, cat, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact dual-chunk position mapping and Neox/non-Neox rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
