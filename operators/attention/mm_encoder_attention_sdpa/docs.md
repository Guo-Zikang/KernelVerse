# mm_encoder_attention_sdpa

## Function

Computes non-causal grouped scaled dot-product attention for multimodal encoder attention.

## Inputs

- `query`: tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- `key`: tensor with shape `[batch_size, num_kv_heads, seq_len, head_dim]`.
- `value`: tensor with shape `[batch_size, num_kv_heads, seq_len, head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- `num_heads` is divisible by `num_kv_heads`.
- Query head `h` uses key/value head `h // (num_heads / num_kv_heads)`.
- This operator is non-causal and does not use an attention mask.
- Use scale factor `head_dim ** -0.5` before softmax.
- Avoid physically materializing repeated K/V heads when practical; map query heads to shared K/V heads directly in the target-backend kernel.
- Implement grouped score computation, stable softmax, value accumulation, and output writeback in backend-native kernel code.
- Do not call PyTorch einsum, matmul, softmax, expand, cat, attention libraries, cuBLAS, CUTLASS, or other operator libraries for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute grouped QK scores, softmax max/sum, and value accumulation in FP32; keep query-head to KV-head mapping exact.
