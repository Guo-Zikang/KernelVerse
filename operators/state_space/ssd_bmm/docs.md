# ssd_bmm

## Function

Computes chunk-local grouped Gram matrices for SSD/Mamba-style state-space blocks.

## Inputs

- `a`: tensor with shape `[total_len, num_groups, key_dim]`.
- `b`: tensor with shape `[total_len, num_groups, key_dim]`.
- `cu_chunk_seqlens`: int32 cumulative chunk offsets with shape `[num_chunks + 1]`.

The selected shape entry provides `seq_lens`; chunks are built independently for each sequence using `chunk_size`. All `a` and `b` values use the dtype selected by the shape entry.

## Output

- `out`: tensor with shape `[num_chunks, num_groups, chunk_size, chunk_size]` and dtype `dtype`.

## Implementation Notes

- For each chunk and group, compute `a_chunk @ b_chunk.T`.
- For incomplete tail chunks, only the valid `[chunk_len, chunk_len]` region is written.
- If `causal` is true, keep only the lower-triangular part of each chunk-local matrix.
- Implement the chunked grouped matrix products and optional causal masking in backend-native kernel code.
- For FP16/BF16 inputs, accumulate each chunk-local `a_chunk @ b_chunk.T` dot product in FP32.
- Apply causal masking after computing valid FP32 products, then cast to the output dtype only when writing `out`.
- Do not call PyTorch matmul, bmm, einsum, triangular mask helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.
