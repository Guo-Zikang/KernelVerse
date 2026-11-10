# reorg_kvcache

## Function

Reorganizes all-gathered distributed KV cache chunks into contiguous local context order.

## Inputs

- `allgathered_kv_c_normed`: tensor with shape `[world_size * total_tokens, feature_dim]`.
- `allgathered_k_pe`: tensor with shape `[world_size * total_tokens, feature_dim]`.
- `padded_local_chunk_seq_lens`: INT32 tensor with shape `[num_sequences]`.
- `local_context_lens_allranks`: INT32 tensor with shape `[num_sequences, world_size + 2]`.

The last two columns of `local_context_lens_allranks` store `chunk_idx` and `chunk_size`.

## Output

Returns a tuple:

- reordered `kv_c`
- reordered `k_pe`

Both outputs have shape `[num_valid_tokens, feature_dim]` and dtype `dtype`.

## Implementation Notes

- This is distributed KV cache data movement, not attention compute.
- For each sequence and rank, copy the valid part of that rank's current chunk.
- Valid chunk length is `min(max(local_context_len - chunk_idx * chunk_size, 0), padded_len)`.
- Source rows are laid out rank-major over the all-gathered input.
- Preserve sequence order, then rank order within each sequence.
- Implement valid-length calculation, segmented copy, and output compaction in backend-native kernel code.
- Do not call PyTorch cat, slicing helpers, boolean indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is segmented cache data movement; preserve copied values exactly in the configured dtype.
- Keep valid-token length, rank/chunk offsets, and output compaction logic exact in integer arithmetic.
- Do not rescale, normalize, or otherwise modify KV feature values during reorganization.
