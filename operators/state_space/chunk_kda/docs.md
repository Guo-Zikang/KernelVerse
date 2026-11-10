# chunk_kda

## Function

Runs chunked KDA attention over a token sequence and returns both token outputs and the final recurrent state.

## Inputs

- `q`: query tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `k`: key tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `v`: value tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `g`: additive gate tensor with shape `[batch_size, seq_len, num_heads]`.
- `beta`: update strength tensor with shape `[batch_size, seq_len, num_heads]`.
- `initial_state`: recurrent state tensor with shape `[batch_size, num_heads, value_dim, key_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `outputs`: tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `final_state`: tensor with shape `[batch_size, num_heads, value_dim, key_dim]`.

Both outputs use the configured shape dtype.

## Implementation Notes

- Implement the chunk-local gate accumulation, KKT score construction, recurrent state update, causal attention path, and final output combination in backend-native kernel code.
- For FP16/BF16 inputs, compute `exp(g)`, beta scaling, KKT scores, recurrent state updates, attention/readout dot products, and output combination in FP32.
- Build chunk outputs and final recurrent state from unrounded FP32 intermediates; cast only when writing final outputs.
- Do not call `torch.cumsum`, `torch.einsum`, `torch.softmax`, matrix libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.
