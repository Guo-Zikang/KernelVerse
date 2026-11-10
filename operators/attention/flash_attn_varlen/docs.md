# flash_attn_varlen

## Function

Computes scaled dot-product attention over packed variable-length Q/K/V buffers. Each sequence slice is defined by cumulative sequence length tensors, and attention is computed independently within each slice.

## Inputs

- `q`: query tensor with shape `[total_tokens, num_heads, head_dim]`.
- `k`: key tensor with shape `[total_tokens, num_heads, head_dim]`.
- `v`: value tensor with shape `[total_tokens, num_heads, head_dim]`.
- `cu_seqlens_q`: int32 cumulative query sequence offsets with shape `[batch_size + 1]`.
- `cu_seqlens_k`: int32 cumulative key/value sequence offsets with shape `[batch_size + 1]`.

The selected shape entry provides `seq_lens`; `total_tokens` is the sum of those lengths. All Q/K/V tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `output`: packed attention output with shape `[total_tokens, num_heads, head_dim]`.
- `softmax_lse`: log-sum-exp tensor with shape `[num_heads, total_tokens]`.

The output tensor uses the configured shape dtype. `softmax_lse` is returned as FP32.

## Implementation Notes

- Implement packed variable-length attention directly from `cu_seqlens_q` and `cu_seqlens_k`.
- Use the scale factor from `softmax_scale` when provided; otherwise use `head_dim ** -0.5`.
- Do not call PyTorch attention, matmul, softmax, exp, einsum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, keep online softmax running max, exp sum, LSE, and value accumulator in FP32 for each variable-length sequence.
