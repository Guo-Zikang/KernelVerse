# recurrent_gated_delta_rule

## Function

Runs recurrent gated delta-rule state updates over a token sequence.

## Inputs

- `q`: query tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `k`: key tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `v`: value tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `beta`: update strength tensor with shape `[batch_size, seq_len, num_heads]`.
- `gate_logits`: sigmoid gate logits with shape `[batch_size, seq_len, num_heads]`.
- `initial_state`: recurrent state tensor with shape `[batch_size, num_heads, value_dim, key_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `outputs`: tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `final_state`: tensor with shape `[batch_size, num_heads, value_dim, key_dim]`.

Both outputs use the configured shape dtype.

## Implementation Notes

- Implement the recurrent state update and query readout in backend-native kernel code.
- Each timestep computes a sigmoid-gated state decay, a beta-scaled key-value update, and a query read from the updated state.
- For FP16/BF16 inputs, compute sigmoid gates, beta-scaled updates, recurrent state updates, and query readout dot products in FP32.
- Do not store recurrent state in FP16/BF16 between timesteps; cast `outputs` and `final_state` only when writing outputs.
- Do not call `torch.sigmoid`, `torch.einsum`, matrix libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.
