# paged_attention_v1

## Function

Computes decode-time paged attention over a KV cache stored as physical blocks.

This operator models grouped-query attention: multiple query heads may share one KV head.

## Inputs

- `query`: tensor with shape `[batch_size, num_q_heads, head_dim]`.
- `key_cache`: tensor with shape `[total_blocks, num_kv_heads, block_size, head_dim]`.
- `value_cache`: tensor with shape `[total_blocks, num_kv_heads, block_size, head_dim]`.
- `seq_lens`: int32 tensor with shape `[batch_size]`.
- `block_table`: int32 tensor with shape `[batch_size, blocks_per_seq]`, mapping logical sequence blocks to physical cache blocks.

All floating input tensors use `input_dtype` from the shape entry.

## Output

- Tensor with shape `[batch_size, num_q_heads, head_dim]`.
- Output dtype must match `output_dtype`.

## Implementation Notes

- For each batch item, use `block_table[batch]` to gather the physical KV blocks for that sequence, flatten the selected blocks in logical order, and truncate to `seq_lens[batch]`.
- For each query head, map to KV head by `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Compute logits as `dot(key, query) * softmax_scale`, where `softmax_scale` defaults to `head_dim ** -0.5`.
- Apply softmax over the valid sequence length and produce the weighted sum over values.
- A high-performance target-backend implementation should avoid physically materializing the gathered full KV sequence when practical and should read cache blocks through `block_table` directly.
- Do not call PyTorch attention, matmul, softmax, indexing/gather, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute paged-cache logits, softmax max/sum, and value accumulation in FP32; block-table lookup and valid-length truncation must stay exact.
