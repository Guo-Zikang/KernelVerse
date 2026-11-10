# chunk_gated_delta_rule_fwd_h

## Function

Computes the recurrent state sequence and updated value tensor used by chunked gated delta-rule attention.

## Inputs

- `k`: key tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `w`: precomputed update tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `u`: value update tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `g`: non-positive log-decay gate tensor with shape `[batch_size, seq_len, num_heads]`.
- `initial_state`: recurrent state tensor with shape `[batch_size, num_heads, value_dim, key_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `h_seq`: recurrent state sequence with shape `[batch_size, seq_len, num_heads, value_dim, key_dim]`.
- `v_new`: updated value tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `final_state`: final recurrent state with shape `[batch_size, num_heads, value_dim, key_dim]`.

Outputs use the configured shape dtype.

## Implementation Notes

- Implement the recurrent gated state update in backend-native kernel code.
- Each timestep computes `state = state * exp(g) + u * k`.
- Each timestep writes the full state to `h_seq` and computes `v_new = u + w`.
- For FP16/BF16 inputs, compute `exp(g)`, the recurrent state update, and `v_new` in FP32.
- Do not store recurrent state in FP16/BF16 between timesteps; cast `h_seq`, `v_new`, and `final_state` only when writing outputs.
- Do not call `torch.exp` or other PyTorch operators for the core computation in `ModelNew.forward`.
