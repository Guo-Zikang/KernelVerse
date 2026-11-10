# mm_encoder_attention_varlen

## Function

Runs non-causal packed variable-length multimodal encoder attention and returns both output and log-sum-exp.

## Inputs

- `query`: packed query tensor with shape `[total_tokens, num_heads, head_dim]`.
- `key`: packed key tensor with shape `[total_tokens, num_heads, head_dim]`.
- `value`: packed value tensor with shape `[total_tokens, num_heads, head_dim]`.
- `cu_seqlens`: cumulative sequence lengths with shape `[num_sequences + 1]`.

Floating tensors use the dtype selected by the shape entry. Sequence metadata uses integer dtype.

## Output

Returns a tuple:

- `output`: tensor with shape `[total_tokens, num_heads, head_dim]`.
- `lse`: tensor with shape `[total_tokens, num_heads]`.

`output` uses the configured shape dtype. `lse` uses FP32.

## Implementation Notes

- This reference path is non-causal.
- Process each packed sequence independently using `cu_seqlens`.
- Use scale factor `head_dim ** -0.5` before softmax.
- Compute softmax with a stable max-subtraction path and return per-query/head log-sum-exp.
- Implement varlen slicing, score computation, softmax, value accumulation, and LSE writeback in backend-native kernel code.
- Do not call PyTorch einsum, matmul, softmax, exp, log, cat, permute, attention libraries, cuBLAS, CUTLASS, or other operator libraries for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, use FP32 online softmax state for each `cu_seqlens` slice: running max, exp sum, LSE, and value accumulator.
