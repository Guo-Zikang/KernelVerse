# kda_output_combine

## Function

Combines KDA attention output with gated value and beta-mixed projection terms.

## Inputs

- `q_proj_states`: query projection tensor with shape `[num_tokens, hidden_dim]`.
- `k_proj_states`: key projection tensor with shape `[num_tokens, hidden_dim]`.
- `v_proj_states`: value projection tensor with shape `[num_tokens, hidden_dim]`.
- `g1`: gate logits tensor with shape `[num_tokens, hidden_dim]`.
- `beta`: beta coefficient tensor with shape `[num_tokens]`.
- `core_attn_out`: core attention output tensor with shape `[num_tokens, hidden_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns one tensor with shape `[num_tokens, hidden_dim]` and the configured shape dtype.

## Implementation Notes

- Implement the sigmoid gate, gated value branch, beta-mix branch, and final add in backend-native kernel code.
- The elementwise formula is `core_attn_out + sigmoid(g1) * v_proj_states + beta * (q_proj_states + k_proj_states) * 0.5`.
- For FP16/BF16 inputs, compute `sigmoid(g1)`, beta-mix terms, gated value branch, and final addition in FP32.
- Cast to the output dtype only when writing the final output tensor.
- Do not call `torch.sigmoid` or other PyTorch operators for the core computation in `ModelNew.forward`.
