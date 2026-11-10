# selective_state_update

## Function

Updates the recurrent state for a Mamba-style selective scan using one step or a small batch of decode tokens.

## Inputs

- `state`: tensor with shape `[num_tokens, num_heads, head_dim, dstate]`.
- `x`: tensor with shape `[num_tokens, num_heads, head_dim]`.
- `dt`: tensor with shape `[num_tokens, num_heads, head_dim]`.
- `A`: tensor with shape `[num_heads, head_dim, dstate]`.
- `B`: tensor with shape `[num_tokens, num_heads, dstate]`.
- `C`: tensor with shape `[num_tokens, num_heads, dstate]`.
- `D`: tensor with shape `[num_heads, head_dim]`.
- `dt_bias`: tensor with shape `[num_heads, head_dim]`.
- `z`: tensor with shape `[num_tokens, num_heads, head_dim]`.

## Output

Returns a tuple:

- `out`: tensor with shape `[num_tokens, num_heads, head_dim]` and dtype `dtype`.
- `new_state`: tensor with shape `[num_tokens, num_heads, head_dim, dstate]` and dtype `dtype`.

## Implementation Notes

- This is a selective scan state update, not a full-sequence scan.
- Compute `dt_eff = dt + dt_bias`; if `dt_softplus` is true, apply softplus to `dt_eff`.
- Update state with `new_state = state * exp(A * dt_eff) + B * dt_eff * x`.
- Project `new_state` with `C`, add skip connection `x * D`, and optionally apply the Swish gate `z * sigmoid(z)`.
- Implement the state update, projection, optional skip, and optional gate in backend-native kernel code.
- For FP16/BF16 inputs, compute softplus, sigmoid/Swish, exponentials, state update, projection accumulation, optional skip, and optional gate in FP32.
- Keep `new_state` in FP32 through projection and cast `out` and `new_state` only when writing outputs.
- Do not call PyTorch exp, softplus, sigmoid, einsum, or other PyTorch operators for the core computation in `ModelNew.forward`.
