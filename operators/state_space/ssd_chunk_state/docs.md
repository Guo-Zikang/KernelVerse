# ssd_chunk_state

## Function

Computes SSD/Mamba-style chunk states from input activations, delta values, A parameters, and B parameters.

## Inputs

- `B`: tensor with shape `[total_len, num_groups, dstate]`.
- `x`: tensor with shape `[total_len, num_heads, head_dim]`.
- `dt`: tensor with shape `[total_len, num_heads]`.
- `A`: tensor with shape `[num_heads]`.
- `cu_chunk_seqlens`: int32 cumulative chunk offsets with shape `[num_chunks + 1]`.
- `dt_bias`: tensor with shape `[num_heads]`.

The selected shape entry provides `seq_lens`; chunks are built independently for each sequence using `chunk_size`. All floating tensors use the dtype selected by the shape entry.

## Output

- `states`: tensor with shape `[num_chunks, num_heads, head_dim, dstate]` and dtype `dtype`.

## Implementation Notes

- This is the SSD chunk-state stage, not the full chunk scan.
- Prepare per-chunk `dt` values and cumulative `A * dt` decay.
- For each chunk and head, accumulate decayed outer products between `x` and grouped `B`.
- `num_heads` is divisible by `num_groups`; head `h` maps to group `h // (num_heads / num_groups)`.
- If `has_dt_bias` is false, ignore `dt_bias`.
- Implement the delta preparation and chunk-state accumulation in backend-native kernel code.
- For FP16/BF16 inputs, compute softplus/clamp if present, `A * dt`, cumulative decay, and decayed outer-product accumulation in FP32.
- Keep chunk-state accumulators in FP32 through the full chunk; cast `states` only when writing the final output.
- Do not call PyTorch matmul, softplus, exp, cumsum, or other PyTorch operators for the core computation in `ModelNew.forward`.
