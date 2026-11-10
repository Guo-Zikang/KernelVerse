# mamba_chunk_scan_combined_varlen

## Function

Runs a variable-length chunked Mamba selective state-space scan and returns token outputs plus final recurrent states.

The operator combines several phases: time-delta preprocessing, per-chunk state contribution, recurrent state propagation across chunks, local chunk scan output, optional skip connection, and optional gate.

## Inputs

- `x`: input tensor with shape `[total_tokens, num_heads, head_dim]`.
- `dt`: time-delta tensor with shape `[total_tokens, num_heads]`.
- `A`: per-head decay tensor with shape `[num_heads]`.
- `B`: state input tensor with shape `[total_tokens, num_groups, dstate]`.
- `C`: state output tensor with shape `[total_tokens, num_groups, dstate]`.
- `cu_seqlens`: cumulative sequence lengths with shape `[num_sequences + 1]`.
- `cu_chunk_seqlens`: cumulative chunk token lengths with shape `[num_chunks + 1]`.
- `last_chunk_indices`: last chunk index for each sequence with shape `[num_sequences]`.
- `D`: optional skip tensor with shape `[num_heads, head_dim]`.
- `z`: optional gate tensor with shape `[total_tokens, num_heads, head_dim]`.
- `dt_bias`: optional time-delta bias tensor with shape `[num_heads]`.
- `initial_states`: optional initial recurrent states with shape `[num_sequences, num_heads, head_dim, dstate]`.

Floating input tensors use the dtype selected by the shape entry. Metadata tensors use integer dtypes.

## Output

Returns a tuple:

- `out`: tensor with shape `[total_tokens, num_heads, head_dim]`.
- `final_states`: tensor with shape `[num_sequences, num_heads, head_dim, dstate]`.

Both floating outputs use the configured shape dtype.

## Implementation Notes

- `seq_lens` may contain different sequence lengths; use `cu_seqlens`, `cu_chunk_seqlens`, and `last_chunk_indices` to respect sequence boundaries.
- If `dt_softplus` is true, apply softplus to `dt + dt_bias` before clamping to non-negative values.
- If `has_dt_bias`, `has_D`, `has_z`, or `has_initial_states` is false, ignore the corresponding tensor argument while keeping the forward signature unchanged.
- `num_heads` is divisible by `num_groups`; each group serves `num_heads / num_groups` heads.
- Implement chunk-local cumulative decay, state accumulation, cross-chunk recurrent state propagation, local scan output, optional skip, and optional gate in backend-native kernel code.
- For FP16/BF16 inputs, compute softplus, sigmoid/Swish, exponentials, chunk-local scan state, cross-chunk recurrent state, and output projection accumulations in FP32.
- Preserve sequence and chunk boundaries exactly; reset FP32 scan state at each new sequence and do not mix carry values across sequences.
- Do not call PyTorch cumsum, einsum, matmul, softplus, silu, exp, or other PyTorch operators for the core computation in `ModelNew.forward`.
