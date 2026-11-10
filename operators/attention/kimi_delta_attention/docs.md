# kimi_delta_attention

## Function

Computes Kimi-style delta attention with learned Q, K, V, alpha, beta, and output projections.

This is a causal recurrent operator that decays the per-head matrix state with a vector alpha gate before applying a vector beta delta-rule update.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V/alpha/beta projections, feature-map application, recurrent alpha decay, beta delta-rule state update, per-token output computation, and output projection in backend-native kernel code.
- The feature map is selected by `feature_map`: `elu` means `elu(x) + 1`, `relu` means ReLU, and `identity` leaves the tensor unchanged.
- Apply the scale factor `head_dim ** -0.5` to both mapped Q and mapped K.
- Accumulate the recurrent state in FP32 for numerical stability.
- For each token `t`, compute `predicted_v = k_t @ state`, `delta_t = v_t - predicted_v`, apply `state = alpha_t[:, None] * state`, update `state += outer(k_t, beta_t * delta_t)`, then output `q_t @ state`.
- Do not call PyTorch attention, matmul, einsum, stack, sigmoid, activation, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, keep alpha/beta gates, delta values, recurrent state decay/update, and `q_t @ state` accumulation in FP32 before casting the final output.
