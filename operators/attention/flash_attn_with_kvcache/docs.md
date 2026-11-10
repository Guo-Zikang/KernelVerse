# flash_attn_with_kvcache

## Function

Computes scaled dot-product attention from query tokens over a key/value cache and returns both attention output and log-sum-exp values.

## Inputs

- `q`: query tensor with shape `[batch_size, seq_len_q, num_heads, head_dim]`.
- `k_cache`: key cache tensor with shape `[batch_size, seq_len_k, num_heads, head_dim]`.
- `v_cache`: value cache tensor with shape `[batch_size, seq_len_k, num_heads, head_dim]`.
- `cache_seqlens`: int32 tensor with shape `[batch_size]`.

All Q/K/V tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `output`: attention output with shape `[batch_size, seq_len_q, num_heads, head_dim]`.
- `softmax_lse`: log-sum-exp tensor with shape `[batch_size, num_heads, seq_len_q]`.

The output tensor uses the configured shape dtype. `softmax_lse` is returned as FP32.

## Implementation Notes

- Use `cache_seqlens` to select the valid prefix of the KV cache for each batch item.
- Use the scale factor from `softmax_scale` when provided; otherwise use `head_dim ** -0.5`.
- Do not call PyTorch attention, matmul, softmax, exp, einsum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute cache attention with FP32 online softmax state and return `softmax_lse` from the same FP32 logsumexp path.
