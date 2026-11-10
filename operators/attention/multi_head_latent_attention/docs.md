# multi_head_latent_attention

## Function

Computes DeepSeek-style multi-head latent attention decode over a paged compressed KV cache.

The query attends to reconstructed cached keys/values according to `block_table` and `cache_seqlens`.

## Inputs

- `query`: tensor with shape `[batch_size, query_seq_len, num_q_heads, qk_head_dim]`.
- `kv_cache`: tensor with shape `[num_blocks, block_size, 1, qk_head_dim]`.
- `block_table`: int tensor with shape `[batch_size, max_blocks_per_seq]`.
- `cache_seqlens`: int tensor with shape `[batch_size]`.

All floating input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, query_seq_len, num_q_heads, v_head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement paged cache reconstruction, QK score computation, optional causal masking, softmax, value aggregation, and output generation in backend-native kernel code.
- `block_table[b]` maps logical cache blocks for batch item `b` to physical blocks in `kv_cache`.
- `cache_seqlens[b]` gives the valid KV length for each batch item.
- K uses the first `qk_head_dim` values from the reconstructed cache. V uses the first `v_head_dim` values.
- For causal mode, query positions are aligned to the end of the KV cache: absolute query position is `query_index + cache_seq_len - query_seq_len`.
- Use `softmax_scale` when present, otherwise use `qk_head_dim ** -0.5`.
- Do not call PyTorch attention, matmul, softmax, cat, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, reconstruct paged latent K/V addresses exactly and compute QK scores, softmax max/sum, and value accumulation in FP32.
