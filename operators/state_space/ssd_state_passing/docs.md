# ssd_state_passing

## Function

Propagates SSD/Mamba-style recurrent states across sequence chunks using per-chunk decay factors.

## Inputs

- `states`: tensor with shape `[num_chunks, num_heads, state_dim]`.
- `dA_cumsum`: tensor with shape `[num_heads, num_chunks, chunk_size]`.
- `last_chunk_indices`: int32 tensor with shape `[batch_size]`; each value is the final chunk index for one sequence.
- `initial_states`: tensor with shape `[batch_size, num_heads, state_dim]`.

All floating tensors use the dtype selected by the shape entry.

## Output

- `out`: tensor with shape `[num_chunks, num_heads, state_dim]` and dtype `dtype`.

## Implementation Notes

- This is the SSD state-passing stage, not the full chunk scan.
- For each sequence, carry a recurrent state across its chunks.
- For chunk `c`, update `prev = exp(dA_cumsum[:, c, -1]) * prev + states[c]`.
- If `has_initial_states` is false, initialize `prev` as zeros for each sequence.
- Implement the chunk-wise recurrence in backend-native kernel code.
- For FP16/BF16 inputs, compute `exp(dA_cumsum[:, c, -1])` and keep the recurrent `prev` state in FP32 across chunks.
- Do not store decay factors or recurrent state in FP16/BF16 between chunks; cast each output chunk to the required dtype only when writing `out`.
- Do not call PyTorch exp or other PyTorch operators for the core computation in `ModelNew.forward`.
