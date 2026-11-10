# top_p_renorm_probs

## Function

Keeps probabilities inside the nucleus/top-p set for each row and renormalizes the remaining probability mass.

## Inputs

- `probs`: probability tensor with shape `[batch_size, vocab_size]`.

All input probabilities use the dtype selected by the shape entry.

## Output

- `probs`: tensor with shape `[batch_size, vocab_size]` and dtype `dtype`.
- Values outside the per-row top-p nucleus are zero.
- Remaining values are renormalized so each row sums to one.

## Implementation Notes

- Input values are probabilities, not logits.
- The reference sorts probabilities in ascending order and keeps the tail whose cumulative mass covers `top_p`.
- Implement threshold selection, mask scatter, row sum, and renormalization in backend-native kernel code.
- Do not call PyTorch sort, cumsum, scatter, where, sum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 probabilities, compute cumulative sums, row sums, and renormalization denominators in FP32.
- Preserve the reference boundary semantics for the top-p threshold.
- Cast only the final renormalized probabilities back to the configured dtype.
