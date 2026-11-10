# moe_unpermute

## Function

Restores expert-contiguous MoE outputs to token top-k order, applies routing weights, and combines routed outputs per token.

## Inputs

- `permuted_hidden_states`: tensor with shape `[num_tokens * topk, hidden_dim]`.
- `topk_weights`: tensor with shape `[num_tokens, topk]`.
- `inv_permuted_idx`: integer tensor with shape `[num_tokens * topk]`.

## Output

Returns one tensor with shape `[num_tokens, hidden_dim]` and dtype `dtype`.

## Implementation Notes

- This is the inverse data movement for `moe_permute` followed by top-k weighted combine.
- `inv_permuted_idx` maps restored flattened route slots back into expert-contiguous rows.
- Restore into logical `[num_tokens, topk, hidden_dim]` order before applying `topk_weights`.
- Apply `topk_weights` along the `topk` dimension and reduce routed expert outputs per token.
- Implement index restore, weighted combine, and reduction in backend-native kernel code.
- Do not call PyTorch index_select, gather/scatter helpers, sum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert restored expert outputs and `topk_weights` to FP32 before multiplication.
- Accumulate routed top-k expert contributions in FP32.
- Cast only the final combined output back to the configured dtype.
