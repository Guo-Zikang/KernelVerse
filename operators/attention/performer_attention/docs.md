# performer_attention

## Function

Computes Performer-style approximate attention with learned Q, K, V, and output projections.

It uses positive random features to approximate the softmax attention kernel, avoiding explicit quadratic attention over all query-key pairs.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, random-feature kernel mapping, linear attention aggregation, normalization, and output projection in backend-native kernel code.
- `num_features` is the number of random feature rows in the projection matrix.
- `random_seed` fixes the orthogonal projection matrix generated from QR blocks in the reference.
- `eps` is used to keep positive random features and denominator normalization numerically stable.
- The reference uses the positive random feature form: scale Q/K by `head_dim ** -0.25`, project with the fixed matrix, subtract half squared norm, exponentiate, and multiply by `num_features ** -0.5`.
- Do not replace this with exact softmax attention; implement the Performer approximation semantics.
- Do not call PyTorch attention, matmul, einsum, exp, linear, QR/random generation, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute random-feature projection, exponentials, `K'^T V`, normalizer, and output division in FP32 before casting the final output.
