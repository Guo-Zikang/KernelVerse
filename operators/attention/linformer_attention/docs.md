# linformer_attention

## Function

Computes Linformer-style attention with learned Q, K, V, output projections, and learned sequence-dimension projections for K and V.

Linformer reduces the key/value sequence length from `seq_len` to `projected_seq_len` before computing attention.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, sequence projections for K/V, scaled dot-product attention over the projected sequence dimension, softmax, value aggregation, and output projection in backend-native kernel code.
- `projected_seq_len` is the compressed K/V length and is always less than or equal to `seq_len`.
- When `share_kv_projection` is true, K and V use the same sequence projection weights.
- Do not replace this with full standard attention over the original sequence length; preserve the Linformer projected K/V semantics.
- Do not call PyTorch attention, linear, matmul, softmax, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute K/V sequence projection, compressed QK scores, softmax max/sum, and projected-value accumulation in FP32 before final output cast.
