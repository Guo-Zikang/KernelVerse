# ssd_chunk_scan

## Function

Runs an SSD/Mamba-style chunk scan pipeline over packed variable-length sequences.

## Inputs

- `x`: tensor with shape `[total_len, num_heads, head_dim]`.
- `dt`: tensor with shape `[total_len, num_heads]`.
- `A`: tensor with shape `[num_heads]`.
- `B`: tensor with shape `[total_len, num_groups, dstate]`.
- `C`: tensor with shape `[total_len, num_groups, dstate]`.
- `cu_chunk_seqlens`: int32 cumulative chunk offsets with shape `[num_chunks + 1]`.
- `last_chunk_indices`: int32 tensor with shape `[num_sequences]`.
- `D`: tensor with shape `[num_heads, head_dim]`.
- `z`: tensor with shape `[total_len, num_heads, head_dim]`.
- `dt_bias`: tensor with shape `[num_heads]`.

The selected shape entry provides `seq_lens`; chunks are built independently for each sequence using `chunk_size`. All floating tensors use the dtype selected by the shape entry.

## Output

- `out`: tensor with shape `[total_len, num_heads, head_dim]` and dtype `dtype`.

## Implementation Notes

- This is a full SSD chunk scan pipeline, not a standalone BMM.
- Compute per-chunk `dt` values, cumulative `A * dt` decay, chunk states, state passing across chunks, local `C @ B` products, and final output combination.
- `num_heads` is divisible by `num_groups`; head `h` maps to group `h // (num_heads / num_groups)`.
- If `has_dt_bias` is false, ignore `dt_bias`.
- If `has_D` is false, ignore the skip connection.
- If `has_z` is false, ignore the Swish gate.
- Implement the chunk scan and optional paths in backend-native kernel code.
- For FP16/BF16 inputs, compute `dt` preprocessing, cumulative `A * dt` decay, chunk states, state passing, local products, optional skip, optional gate, and final output accumulation in FP32.
- Preserve sequence and chunk boundaries exactly; keep cross-chunk carry/state values in FP32 and cast only when writing final outputs.
- Do not call PyTorch matmul, einsum, softplus, sigmoid, exp, cumsum, or other PyTorch operators for the core computation in `ModelNew.forward`.
