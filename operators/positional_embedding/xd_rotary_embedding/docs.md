# xd_rotary_embedding

## Function

Applies sectioned multi-axis XD rotary embeddings to query and key tensors.

## Inputs

- `positions`: integer position tensor with shape `[num_axes, batch_size * seq_len]`.
- `query`: tensor with shape `[batch_size * seq_len, num_q_heads * head_size]`.
- `key`: tensor with shape `[batch_size * seq_len, num_kv_heads * head_size]`.

`query` and `key` use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `query_out`: tensor with the same shape and dtype as `query`.
- `key_out`: tensor with the same shape and dtype as `key`.

## Implementation Notes

- Split `rotary_dim // 2` into axis sections according to `xdrope_section`.
- Build separate cosine and sine tables for each axis in `positions`.
- Concatenate the axis-wise cosine and sine sections, then rotate the first `rotary_dim` channels of each head.
- Preserve the channels after `rotary_dim` unchanged.
- Support both NeoX split-half layout and interleaved pair layout through `is_neox_style`.
- Do not call PyTorch rotary helper functions or other existing RoPE implementations for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact axis-wise frequency, angle, sin/cos, and rotation path.
- Preserve the exact `xdrope_section` mapping and NeoX/interleaved rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
