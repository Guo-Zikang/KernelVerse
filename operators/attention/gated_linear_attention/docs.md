# gated_linear_attention

## Function

Computes recurrent gated linear attention with learned Q, K, V, gate, and output projections.

This is a causal recurrent operator that applies a vector gate to decay the per-head matrix state before adding the current key/value outer product.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V/gate projections, feature-map application, recurrent gated state update, per-token output computation, and output projection in backend-native kernel code.
- The feature map is selected by `feature_map`: `elu` means `elu(x) + 1`, `relu` means ReLU, and `identity` leaves the tensor unchanged.
- For each token `t`, compute `kv_t = outer(k_t, v_t)`, update `state = gate_t[:, None] * state + kv_t`, then output `q_t @ state`.
- `gate_t` is produced by a sigmoid projection and has shape `[batch_size, num_heads, head_dim]`.
- Do not call PyTorch attention, matmul, einsum, stack, sigmoid, activation, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute feature maps, sigmoid gates, recurrent state updates, and `q_t @ state` accumulation in FP32 before casting the final output.
