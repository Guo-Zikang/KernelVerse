# moe_permute

## Function

Permutes MoE token hidden states into expert-contiguous order according to `topk_ids`.

## Inputs

- `hidden_states`: tensor with shape `[num_tokens, hidden_dim]`.
- `topk_ids`: integer tensor with shape `[num_tokens, topk]`.

Each `topk_ids` entry is an expert id in `[0, num_experts)`.

## Output

Returns a tuple:

- `permuted_hidden_states`: hidden states repeated and grouped by routed expert, shape `[num_tokens * topk, hidden_dim]`.
- `expert_first_token_offset`: cumulative expert offsets, shape `[num_experts + 1]`.
- `inv_permuted_idx`: inverse permutation index for restoring expert outputs.
- `permuted_idx`: flattened route indices sorted by expert id.

Metadata outputs use `index_dtype`.

## Implementation Notes

- This is MoE token routing permutation, not a tensor dimension transpose.
- Preserve stable ordering among routes assigned to the same expert.
- `permuted_idx` indexes flattened `[num_tokens, topk]` route slots.
- `inv_permuted_idx[permuted_idx[i]] == i`.
- Implement counting, prefix offsets, route ordering, hidden-state gather, and inverse-index writeback in backend-native kernel code.
- Do not call PyTorch argsort, bincount, cumsum, index_select, gather/scatter helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Preserve integer route ids, offsets, counts, and inverse indices exactly.
- Preserve stable ordering among routes assigned to the same expert.
- Do not change floating hidden-state values during pure permutation.
