# paged_attention_v2

## Function

Computes decode-time paged attention over a KV cache using partition-local softmax statistics and a global merge.

This operator models grouped-query attention: multiple query heads may share one KV head.

## Inputs

- `query`: tensor with shape `[batch_size, num_q_heads, head_dim]`.
- `key_cache`: tensor with shape `[total_blocks, num_kv_heads, block_size, head_dim]`.
- `value_cache`: tensor with shape `[total_blocks, num_kv_heads, block_size, head_dim]`.
- `seq_lens`: int32 tensor with shape `[batch_size]`.
- `block_table`: int32 tensor with shape `[batch_size, blocks_per_seq]`, mapping logical sequence blocks to physical cache blocks.

All floating input tensors use `input_dtype` from the shape entry.

## Output

Returns a tuple:

- `output`: final decode output with shape `[batch_size, num_q_heads, head_dim]`.
- `exp_sums`: per-partition softmax exp sums with shape `[batch_size, num_q_heads, max_num_partitions]`.
- `max_logits`: per-partition softmax max logits with the same shape as `exp_sums`.
- `tmp_out`: per-partition normalized temporary output with shape `[batch_size, num_q_heads, max_num_partitions, head_dim]`.

All returned tensors must use `output_dtype`.

## Implementation Notes

- For each batch item, use `block_table[batch]` to gather logical KV blocks from physical cache blocks and truncate to `seq_lens[batch]`.
- For each query head, map to KV head by `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Split the valid sequence into partitions of `partition_size`; the last partition may be shorter.
- For each partition, compute local logits, local max, exp sum, weighted value sum, and normalized temporary output.
- Merge partitions with the numerically stable global softmax formula using the per-partition max logits and exp sums.
- Preserve the four-output tuple structure exactly.
- A high-performance target-backend implementation should avoid physically materializing the gathered full KV sequence when practical and should read cache blocks through `block_table` directly.
- Do not call PyTorch attention, matmul, softmax, exp, stack, indexing/gather, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute partition max logits, exp sums, weighted values, and global partition merge in FP32; cast only the final output tensor.
