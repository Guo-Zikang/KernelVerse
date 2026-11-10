# block_sparse_attention

## Function

Computes block sparse attention with learned Q, K, V, and output projections.

The sequence is split into fixed-size blocks. For each query block, attention is computed only against selected key/value blocks.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, block selection, sparse attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- `seq_len` is divisible by `block_size`; `num_blocks = seq_len / block_size`.
- `fixed` mode selects consecutive key blocks starting from each query block, wrapping around by `num_blocks`.
- `random` mode uses `random_seed` to create a deterministic per-query-block table of selected key blocks.
- `topk` mode selects key blocks by top-k block summary scores computed from mean-pooled Q/K blocks.
- `local_plus_topk` mode always includes the matching local block and fills the remaining slots with top-k non-local blocks.
- For equal selection scores, use deterministic tie-breaking by smaller block index.
- Avoid materializing dense full-sequence attention. Prefer computing only selected block pairs.
- Do not call PyTorch attention, matmul, topk, softmax, linear, random, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute block summaries, block-score selection, sparse QK scores, softmax max/sum, and probability-value accumulation in FP32; keep deterministic tie-breaking for selected blocks.
