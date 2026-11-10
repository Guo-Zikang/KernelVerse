# nystromformer_attention

## Function

Computes Nyströmformer-style approximate self-attention using landmark pooling and an iterative pseudo-inverse of the landmark attention kernel.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference applies Q/K/V projections, reshapes to `[batch_size, num_heads, seq_len, head_dim]`, and computes either dense attention or the Nyström approximation.
- If `seq_len <= num_landmarks`, the reference uses normal dense softmax attention.
- Otherwise, Q and K are pooled into `num_landmarks` contiguous segments. The final segment may include zero padding before averaging.
- The approximation computes three kernels: query-to-landmark, landmark-to-landmark, and landmark-to-key. It then multiplies `kernel_1 * pseudo_inverse(kernel_2) * (kernel_3 * V)`.
- The pseudo-inverse uses the fixed Newton-Schulz style iteration controlled by `pinv_iterations` and stabilized by `eps`.
- A high-performance target-backend implementation should avoid unnecessary materialization and fuse kernel construction, softmax, pseudo-inverse iteration, and value aggregation when practical.
- Do not call PyTorch attention, matmul, softmax, linear, padding, inverse, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute landmark pooling, kernel softmaxes, pseudo-inverse iteration, and matrix accumulations in FP32 before casting the final output.
