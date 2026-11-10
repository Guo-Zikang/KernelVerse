# token_routing_sparse_attention

## Function

Computes token-level top-k routing sparse self-attention. Tokens can attend to each other when their selected routes overlap.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference computes Q/K/V projections, plus a router projection from token hidden state to `num_routes` logits.
- For each token, select `top_k_routes` route indices from the router logits.
- For equal selection scores, use deterministic tie-breaking by smaller route index.
- Build a sparse visibility mask where query token `i` can attend key token `j` if their selected route sets overlap.
- The diagonal is always visible, even if no selected routes overlap.
- Apply scaled dot-product attention only over visible positions, then apply the output projection `w_o`.
- A high-performance target-backend implementation should avoid materializing full one-hot route membership and full `[seq_len, seq_len]` masks when practical.
- Do not call PyTorch attention, matmul, topk, one_hot, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute router logits, top-k route selection, sparse QK scores, softmax max/sum, and value accumulation in FP32; keep deterministic tie-breaking for selected routes.
