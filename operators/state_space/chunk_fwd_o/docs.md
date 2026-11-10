# chunk_fwd_o

## Function

Computes the output-combine stage used by chunked state-space attention variants.

The operator combines:

- causal attention over the current chunk from `q`, `k`, and `v`;
- recurrent state readout from `h` and `q`;
- sigmoid gating from `g`;
- final output `attention_output + sigmoid(g) * state_output`.

## Inputs

- `q`: query tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `k`: key tensor with shape `[batch_size, seq_len, num_heads, key_dim]`.
- `v`: value tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.
- `h`: recurrent state tensor with shape `[batch_size, seq_len, num_heads, value_dim, key_dim]`.
- `g`: gate tensor with shape `[batch_size, seq_len, num_heads]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns one tensor with shape `[batch_size, seq_len, num_heads, value_dim]`.

The output uses the dtype selected by the shape entry.

## Implementation Notes

- Implement the causal attention path, recurrent state readout, sigmoid gate, and final add in backend-native kernel code.
- For FP16/BF16 inputs, compute QK attention scores, value accumulation, recurrent state readout, sigmoid gate, and final addition in FP32.
- Cast to the output dtype only when writing the final output tensor.
- Do not call `torch.einsum`, `torch.softmax`, matrix libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.
