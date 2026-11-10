# mla_decode

## Function

Runs MLA decode attention over a paged KV cache.

This operator uses DeepSeek-style MLA dimensions:

- latent dimension: `512`
- RoPE dimension: `64`
- total key dimension: `576`
- softmax scale: `1.44`

## Inputs

- `q_nope`: query latent tensor with shape `[batch_size, num_q_heads, 512]`.
- `q_pe`: query RoPE tensor with shape `[batch_size, num_q_heads, 64]`.
- `kv_cache`: paged KV cache tensor with shape `[num_cache_blocks, block_size, 576]`.
- `seq_lens`: valid sequence lengths with shape `[batch_size]`.
- `block_table`: cache block indices with shape `[batch_size, num_blocks]`.

Floating input tensors use the dtype selected by the shape entry. Metadata tensors use integer dtypes.

## Output

- Tensor with shape `[batch_size, num_q_heads, 512]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Concatenate `q_nope` and `q_pe` logically to form a `[num_q_heads, 576]` query per batch item.
- Use `block_table` to gather the valid paged KV-cache blocks for each batch item.
- For each batch item, only attend over the first `seq_lens[batch_idx]` cache rows after flattening gathered blocks.
- Keys are the full `576` cache dimension; values are the first `512` dimensions of the same cache rows.
- `block_size` divides `128`; `num_blocks` is padded to the corresponding packing factor in `generate_inputs`.
- Implement paged cache lookup, attention score computation, softmax, and value accumulation in backend-native kernel code.
- Do not call PyTorch cat, matmul, softmax, indexing helpers, attention libraries, cuBLAS, CUTLASS, or other operator libraries for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the logical Q/K dot product, softmax max/sum, and value accumulation in FP32 while keeping paged-cache address calculations exact.
