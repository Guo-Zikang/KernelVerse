# selective_scan

## Function

Runs a Mamba-style selective scan recurrence over a full sequence.

## Inputs

- `u`: tensor with shape `[batch_size, dim, seqlen]`.
- `delta`: tensor with shape `[batch_size, dim, seqlen]`.
- `A`: tensor with shape `[dim, dstate]`.
- `B`: tensor with shape `[batch_size, n_groups, dstate, seqlen]`.
- `C`: tensor with shape `[batch_size, n_groups, dstate, seqlen]`.
- `D`: tensor with shape `[dim]`.
- `z`: tensor with shape `[batch_size, dim, seqlen]`.
- `delta_bias`: tensor with shape `[dim]`.
- `initial_state`: tensor with shape `[batch_size, dim, dstate]`.
- `has_initial_state`: BOOL tensor with shape `[batch_size]`.

## Output

Returns a tuple:

- `out`: tensor with shape `[batch_size, dim, seqlen]` and dtype `dtype`.
- `state`: tensor with shape `[batch_size, dim, dstate]` and dtype `dtype`.

## Implementation Notes

- This is Mamba selective scan recurrence, not a generic prefix sum.
- For each time step and channel, update recurrent state with `exp(A * delta)`.
- `B` and `C` are grouped by `n_groups`; channel `d` uses group `d // (dim / n_groups)`.
- If `delta_softplus` is true, apply softplus to `delta + delta_bias`.
- If `has_D` is true, add skip connection `u * D`.
- If `has_z` is true, apply the Swish gate `z * sigmoid(z)`.
- If `use_initial_state` is true, initialize state from `initial_state` where `has_initial_state` is true.
- Implement the recurrence, optional skip, optional gate, and final state writeback in backend-native kernel code.
- For FP16/BF16 inputs, compute softplus, sigmoid/Swish, exponentials, recurrent scan state, projection accumulations, optional skip, and optional gate in FP32.
- Do not store recurrent state in FP16/BF16 between timesteps; cast `out` and `state` only when writing outputs.
- Do not call PyTorch exp, softplus, sigmoid, cumulative-scan helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.
