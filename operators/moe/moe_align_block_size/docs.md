# moe_align_block_size

## Function

Sorts flattened MoE top-k token routes by expert and pads each expert bucket to a block-size boundary.

## Inputs

- `topk_ids`: integer tensor with shape `[num_tokens, topk]`.

Each entry is an expert id in `[0, num_experts)`.

## Output

Returns a tuple:

- `sorted_token_ids`: flattened token-route positions sorted by expert and padded to `block_size`.
- `expert_ids`: expert id for each padded block.
- `num_tokens_post_pad`: scalar tensor storing the total padded token-route count.
- `cumsum_buffer`: cumulative padded token-route counts per expert, shape `[num_experts + 1]`.

All outputs use `output_index_dtype`.

## Implementation Notes

- Flatten `topk_ids` before grouping routes by expert.
- The sentinel id for invalid padded entries is `num_tokens * topk`.
- If `pad_sorted_token_ids` is false, padded `sorted_token_ids` entries remain the sentinel.
- If `pad_sorted_token_ids` is true, padded entries repeat the last real token-route id for that expert bucket.
- Empty expert buckets produce no output blocks but still contribute an unchanged cumulative count.
- Implement route counting, expert grouping, block padding, cumulative counts, and metadata writeback in backend-native kernel code.
- Do not call PyTorch nonzero, cat, tensor construction, sorting/grouping helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Preserve integer route ids, expert ids, padded counts, cumulative counts, and sentinel values exactly.
- Preserve stable ordering for routes assigned to the same expert.
- Do not change route metadata dtype or sentinel semantics.
