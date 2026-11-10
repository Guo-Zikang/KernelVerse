# native_sparse_attention

## Function

Computes native sparse attention using three additive branches: compressed block attention, selected raw-block attention, and causal local-window attention.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference creates Q/K/V projections, reshapes them to `[batch_size, num_heads, seq_len, head_dim]`, computes the three sparse attention branches, sums them, and applies the output projection.
- `block_size` partitions the sequence into full blocks. `seq_len` is always divisible by `block_size`.
- For a query position, completed blocks are blocks whose end position is less than or equal to `query_idx + 1`. Future or incomplete blocks must not contribute.
- The compressed branch averages keys and values within each completed block and attends over those compressed block entries.
- The selected branch scores completed compressed keys, selects the top `num_selected_blocks`, then attends over the raw tokens inside the selected blocks.
- For equal selection scores, use deterministic tie-breaking by smaller block index.
- The local branch attends only to causal local positions `[query_idx - local_window_size + 1, query_idx]`, clipped to the beginning of the sequence.
- The PyTorch reference spells out the sparse branches for clarity. A high-performance target-backend implementation should avoid materializing full `[seq_len, seq_len]` attention and compute compressed blocks, selected raw blocks, and local windows directly.
- Do not call PyTorch attention, matmul, topk, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute compressed block summaries, selected-block scores, branch softmax reductions, and value accumulation in FP32; tie-breaking must still prefer smaller block indices.
