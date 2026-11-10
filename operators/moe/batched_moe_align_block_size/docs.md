# batched_moe_align_block_size

## Function

Builds block-aligned metadata for batched MoE expert token buckets.

## Inputs

- `expert_num_tokens`: `INT32` tensor with shape `[num_experts]`.

Each element stores the number of real tokens assigned to one expert bucket. The count is in `[0, max_tokens_per_batch]`.

## Output

Returns a tuple:

- `sorted_ids`: padded token ids for block-based MoE kernels.
- `expert_ids`: expert id for each padded block.
- `num_tokens_post_pad`: scalar tensor storing the total number of tokens after padding.

All outputs use `output_index_dtype`.

## Implementation Notes

- The input is token counts per expert bucket, not top-k routing ids.
- Each expert bucket is padded up to a multiple of `block_size`.
- Empty expert buckets should produce no output blocks.
- Padded `sorted_ids` entries use a sentinel id equal to `num_experts * max_tokens_per_batch`.
- Implement the count alignment and metadata construction in backend-native kernel code.
- Do not call PyTorch allocation/fill/indexing helpers for the core computation in `ModelNew.forward`.

## Precision Notes

- Preserve integer token counts, expert ids, padded counts, and sentinel values exactly.
- Empty expert buckets must keep exact zero-count metadata.
- Do not change metadata dtype or sentinel semantics.
