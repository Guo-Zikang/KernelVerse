# lm_head_logits

## Function

Computes LM head vocabulary logits from hidden states.

## Inputs

- `hidden_states`: tensor with shape `[num_tokens, hidden_dim]`.
- `weight`: tensor with shape `[vocab_size, hidden_dim]`.
- `bias`: tensor with shape `[vocab_size]`, or `None` when `use_bias` is false.

## Output

Returns logits with shape `[num_tokens, vocab_size]` and dtype `dtype`.

## Implementation Notes

- Compute `hidden_states @ weight.T`, optionally adding `bias`.
- Treat `vocab_size` as the output dimension.
- Implement the projection in backend-native kernel code.
- Do not call PyTorch linear, matmul, addmm, GEMM helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision and Performance Notes

- For FP16/BF16 inputs, accumulate the hidden dimension in FP32 or hardware mixed-precision matrix accumulators, then add optional bias and cast once to the output dtype.
- Optimize for large `vocab_size`: tile the output dimension, use coalesced reads from `weight`, and avoid extra temporary logits buffers when possible.
- Use hardware matrix acceleration for aligned FP16/BF16 shapes when practical, with correct fallback for tail and unaligned sizes.
