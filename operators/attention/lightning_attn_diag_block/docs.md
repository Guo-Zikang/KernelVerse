# lightning_attn_diag_block

## Function

Computes decayed intra-block lightning attention. Each sequence block applies causal attention scores with an exponential decay mask based on the relative token distance inside the block.

## Inputs

- `q`: query tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- `k`: key tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- `v`: value tensor with shape `[batch_size, num_heads, seq_len, value_dim]`.
- `s`: per-head decay tensor with shape `[num_heads]`.

All `q`, `k`, and `v` tensors use the dtype selected by the shape entry. `s` uses FP32.

## Output

- Tensor with shape `[batch_size, num_heads, seq_len, value_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- For each block, compute causal decayed scores only within that block.
- The decay factor for output row `i` and key column `j` is `exp(-s[h] * (i - j))` when `i >= j`, otherwise zero.
- Implement the block score computation, decay mask application, and value accumulation in backend-native kernel code.
- Do not call PyTorch attention, matmul, einsum, exp, where, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute decay exponentials, decayed scores, and value accumulation in FP32 within each causal block before casting the final output.
