# lightning_attn_kv_reduce

## Function

Computes one decayed `K^T V` matrix for each sequence block and attention head.

## Inputs

- `k`: key tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- `v`: value tensor with shape `[batch_size, num_heads, seq_len, value_dim]`.
- `k_decay`: per-head decay tensor with shape `[num_heads, block_size]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, num_heads, num_blocks, head_dim, value_dim]`, where `num_blocks = ceil(seq_len / block_size)`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- For each block, multiply each key row by the corresponding `k_decay` value before accumulating `K^T V`.
- The last block may contain fewer than `block_size` tokens.
- Implement the decay application and per-block `K^T V` accumulation in backend-native kernel code.
- Do not call PyTorch matmul, einsum, stack, attention, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, apply `k_decay` and accumulate each per-block `K^T V` result in FP32 before casting the block output to the configured dtype.
