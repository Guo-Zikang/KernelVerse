# recurrent_gated_delta_rule_packed_decode

## Function

Runs packed single-token recurrent gated delta-rule decode updates.

## Inputs

- `mixed`: packed tensor with shape `[batch_size, 1, num_heads, 2 * key_dim + value_dim]`. The last dimension stores `q`, `k`, and `v` in that order.
- `initial_state`: recurrent state tensor with shape `[batch_size, num_heads, value_dim, key_dim]`.
- `ssm_state_indices`: integer tensor selecting which batch states to return.

`mixed` and `initial_state` use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `outputs`: decode output tensor with shape `[batch_size, 1, num_heads, value_dim]`.
- `gathered_state`: selected recurrent states with shape `[batch_size, num_heads, value_dim, key_dim]`.

Floating outputs use the configured shape dtype.

## Implementation Notes

- Implement the packed q/k/v split, single-token recurrent state update, query readout, and state gather in backend-native kernel code.
- Decode uses neutral update strength and gate logits: beta is 1, and the sigmoid gate value is 0.5.
- For FP16/BF16 inputs, compute the single-token state update and query readout dot products in FP32.
- Keep the updated recurrent state in FP32 until writing `outputs` and `gathered_state`.
- Do not call `torch.split`, `torch.einsum`, `index_select`, matrix libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.
