# linear_attention

## Function

Computes non-recurrent linear attention using feature maps and the `K^T V` reordering.

The attention approximation is computed as `(Q (K^T V)) / (Q sum(K) + eps)`.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, feature-map application, `K^T V` accumulation, denominator normalization, output computation, and output projection in backend-native kernel code.
- The feature map is selected by `feature_map`: `elu` means `elu(x) + 1`, `relu` means ReLU, and `identity` leaves the tensor unchanged.
- Use `eps` to stabilize the reciprocal denominator `1 / (q dot sum(k) + eps)`.
- This operator is not recurrent; compute the global `K^T V` and `sum(K)` over the whole sequence.
- Do not call PyTorch attention, matmul, einsum, reciprocal, activation, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute feature maps, `K^T V`, `sum(K)`, denominator normalization, and output accumulation in FP32 before casting the final output.
