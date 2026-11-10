# chunk_cumsum_dt

## Function

Preprocesses variable-length state-space `dt` values into chunked `dt_out` and `dA_cumsum` tensors for Mamba-style scan kernels.

## Inputs

- `dt`: time-step tensor with shape `[total_seq_len, num_heads]`.
- `A`: state transition tensor with shape `[num_heads]`.
- `cu_chunk_seqlens`: cumulative chunk lengths with shape `[num_chunks + 1]`.
- `dt_bias`: optional bias tensor with shape `[num_heads]`; used when `has_dt_bias` is true.

`dt`, `A`, and `dt_bias` use the dtype selected by the shape entry. `cu_chunk_seqlens` is `INT32`.

## Output

Returns a tuple:

- `dA_cumsum`: tensor with shape `[num_heads, num_chunks, chunk_size]`.
- `dt_out`: tensor with shape `[num_chunks, num_heads, chunk_size]`.

Both outputs use the dtype selected by the shape entry.

## Implementation Notes

- This is state-space scan preprocessing, not a generic cumsum.
- Each chunk is processed independently according to `cu_chunk_seqlens`.
- Apply `dt_bias` only when `has_dt_bias` is true.
- Apply softplus only when `dt_softplus` is true, using the same threshold behavior as the reference.
- Clamp processed `dt` values to be non-negative.
- For FP16/BF16 inputs, compute softplus, clamp, `A * dt`, and chunk-local cumulative sums in FP32.
- Do not store cumsum carry values in FP16/BF16 between scan steps; cast `dA_cumsum` and `dt_out` only when writing final outputs.
- Do not call `torch.cumsum`, `torch.where`, `torch.clamp`, PyTorch softplus, or other PyTorch operators for the core computation in `ModelNew.forward`.
