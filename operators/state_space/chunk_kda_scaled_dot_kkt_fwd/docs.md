# chunk_kda_scaled_dot_kkt_fwd

## Function

Computes chunk-local KDA scaled KKT products and their prefix sums.

## Inputs

- `k`: key tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `beta`: update strength tensor with shape `[batch_size, seq_len, num_heads]`.
- `g`: additive gate tensor with shape `[batch_size, seq_len, num_heads]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `A`: chunk-local KKT tensor with shape `[batch_size, seq_len, num_heads, chunk_size]`.
- `Aqk`: prefix sum of `A` along the chunk dimension with shape `[batch_size, seq_len, num_heads, chunk_size]`.

Both outputs use the configured shape dtype.

## Implementation Notes

- Implement the chunk-local KKT product and prefix sum in backend-native kernel code.
- The computation is independent for each chunk.
- For FP16/BF16 inputs, compute `exp(g)`, `k * exp(g)`, beta scaling, KKT dot products, and the `Aqk` prefix sum in FP32.
- Build `Aqk` from the unrounded FP32 KKT values; cast `A` and `Aqk` to the output dtype only when writing the final outputs.
- Do not call `torch.exp`, `torch.einsum`, `torch.cumsum`, matrix libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.
