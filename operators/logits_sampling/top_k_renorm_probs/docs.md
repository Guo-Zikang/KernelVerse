# top_k_renorm_probs

## Function

Keeps probabilities in the top-k set for each row and renormalizes the remaining probability mass.

## Inputs

- `probs`: probability tensor with shape `[batch_size, vocab_size]`.

All input probabilities use the dtype selected by the shape entry.

## Output

- `probs`: tensor with shape `[batch_size, vocab_size]` and dtype `dtype`.
- Values outside the per-row top-k threshold are zero.
- Remaining values are renormalized so each row sums to one.

## Implementation Notes

- Input values are probabilities, not logits.
- The reference keeps entries greater than or equal to the kth largest probability, so ties at the threshold may keep more than `top_k` values.
- Implement threshold selection, masking, row sum, and renormalization in backend-native kernel code.
- Do not call PyTorch sort, topk, where, sum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 probabilities, compute threshold selection, row sums, and renormalization denominators in FP32.
- Preserve the reference boundary semantics for ties at the top-k threshold.
- Cast only the final renormalized probabilities back to the configured dtype.
