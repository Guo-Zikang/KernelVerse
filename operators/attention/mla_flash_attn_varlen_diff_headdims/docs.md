# mla_flash_attn_varlen_diff_headdims

## Function

Runs packed variable-length MLA flash attention where the Q/K head dimension and V head dimension may differ.

## Inputs

- `query`: packed query tensor with shape `[total_q, num_heads, qk_head_dim]`.
- `key`: packed key tensor with shape `[total_k, num_heads, qk_head_dim]`.
- `value`: packed value tensor with shape `[total_k, num_heads, v_head_dim]`.
- `cu_seqlens_q`: cumulative query sequence lengths with shape `[num_sequences + 1]`.
- `cu_seqlens_k`: cumulative key/value sequence lengths with shape `[num_sequences + 1]`.

Floating tensors use the dtype selected by the shape entry. Sequence metadata uses integer dtype.

## Output

Returns a tuple:

- `output`: tensor with shape `[total_q, num_heads, v_head_dim]`.
- `lse`: tensor with shape `[total_q, num_heads]`.

`output` uses the configured shape dtype. `lse` uses FP32.

## Implementation Notes

- Process each packed sequence independently using the prefix sums.
- Use Q/K dimension `qk_head_dim` for scores and V dimension `v_head_dim` for the output.
- If `causal` is true, use the varlen causal mask `k_pos <= q_pos + seq_k - seq_q`.
- Compute softmax with a stable max-subtraction path and return per-query/head log-sum-exp.
- Implement varlen slicing, score computation, causal masking, softmax, value accumulation, and LSE writeback in backend-native kernel code.
- Do not call PyTorch einsum, matmul, softmax, exp, log, cat, permute, attention libraries, cuBLAS, CUTLASS, or other operator libraries for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, keep QK score accumulation, online softmax max/sum, LSE, and value accumulation in FP32 even when QK and V head dimensions differ.
