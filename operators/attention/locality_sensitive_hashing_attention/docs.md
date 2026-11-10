# locality_sensitive_hashing_attention

## Function

Computes locality-sensitive hashing attention with learned Q, K, V, and output projections.

Tokens may attend to each other when they share at least one hash bucket across `num_hashes` hash rounds. Each token always attends to itself.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, LSH bucket assignment, sparse bucket-local attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- `random_rotations` is created from `hash_seed` with shape `[num_hashes, num_heads, head_dim, num_buckets]`.
- For each hash round, bucket id is `argmax(q @ random_rotation)` over `num_buckets`.
- For equal selection scores, use deterministic tie-breaking by smaller bucket index.
- The attention mask is true for a pair of tokens if they share a bucket in any hash round; diagonal self-attention is always true.
- `num_buckets == 1` is the full-attention degenerate case.
- The PyTorch reference materializes the full `[seq_len, seq_len]` mask to make the sparse semantics explicit; a target-backend implementation should hash, sort, and bucket tokens first, then compute attention only inside buckets when practical.
- Do not call PyTorch attention, matmul, einsum, argmax, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute LSH bucket scores, bucket assignment, in-bucket QK scores, softmax max/sum, and value accumulation in FP32; ties should prefer smaller bucket indices.
