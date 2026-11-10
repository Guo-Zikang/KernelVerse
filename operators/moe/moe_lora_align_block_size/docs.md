# moe_lora_align_block_size

## Function

Aligns MoE top-k token routes into per-LoRA-adapter expert blocks.

## Inputs

- `topk_ids`: integer tensor with shape `[num_tokens, topk]`.
- `token_lora_mapping`: integer tensor with shape `[num_tokens]`.

Each `topk_ids` entry is an expert id in `[0, num_experts)`. Each token is assigned to one LoRA adapter id in `[0, max_loras)`.

## Output

Returns a tuple:

- `sorted_token_ids`: per-adapter routed token ids padded to block-size boundaries, shape `[max_loras, max_num_tokens_padded]`.
- `expert_ids`: expert id for each padded block, shape `[max_loras, max_num_m_blocks]`.
- `num_tokens_post_pad`: per-adapter padded token-route count, shape `[max_loras]`.
- `adapter_enabled`: per-adapter flag indicating whether an adapter appears in the current token batch, shape `[max_loras]`.
- `lora_ids`: adapter id stored for each active output slot, shape `[max_loras]`.

All outputs use `output_index_dtype`.

## Implementation Notes

- This differs from `moe_align_block_size` because routing is grouped by LoRA adapter first, then by expert inside each adapter.
- Flatten `topk_ids` before grouping routes.
- The sentinel id for invalid padded entries is `num_tokens`.
- Adapter output slots follow sorted active LoRA ids.
- Implement adapter grouping, expert grouping, block padding, and metadata writeback in backend-native kernel code.
- Do not call PyTorch unique, nonzero, tensor construction, sorting/grouping helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Preserve integer route ids, adapter ids, expert ids, padded counts, and sentinel values exactly.
- Preserve stable ordering inside each LoRA adapter and expert bucket.
- Do not change metadata dtype or sentinel semantics.
