# yarn_scaling_rotary_embedding

## Function

Applies YaRN-scaled rotary embeddings to query and key tensors for long-context position encoding.

## Inputs

- `positions`: integer position tensor with shape `[batch_size * seq_len]`.
- `query`: tensor with shape `[batch_size * seq_len, num_q_heads * head_size]`.
- `key`: tensor with shape `[batch_size * seq_len, num_kv_heads * head_size]`.

`query` and `key` use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `query_out`: tensor with the same shape and dtype as `query`.
- `key_out`: tensor with the same shape and dtype as `key`.

## Implementation Notes

- Compute YaRN inverse frequencies by blending interpolated and extrapolated frequencies.
- Use `beta_fast`, `beta_slow`, `max_position_embeddings`, and `truncate` to build the correction range and ramp mask.
- Multiply cosine and sine tables by the YaRN magnitude scale derived from `scaling_factor` and `attn_factor`.
- Rotate the first `rotary_dim` channels of each head and preserve channels after `rotary_dim`.
- Support both NeoX split-half layout and interleaved pair layout through `is_neox_style`.
- Do not call PyTorch rotary helper functions or other existing RoPE implementations for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact frequency, angle, sin/cos, and rotation path.
- Preserve the exact YaRN scaling formula and NeoX/interleaved rotary layout; large errors usually come from phase or layout mismatches rather than ordinary accumulation error.
