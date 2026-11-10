# logits_processor_scale_softcap

## Function

Applies optional soft capping and scaling to logits before sampling.

## Inputs

- `logits`: tensor with shape `[num_seqs, vocab_size]`.

The input tensor uses the dtype selected by the shape entry.

## Output

- Tensor with shape `[num_seqs, vocab_size]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- If `soft_cap` is not `None`, compute `tanh(logits / soft_cap) * soft_cap`.
- If `scale != 1.0`, multiply the result by `scale`.
- Implement the tanh, scaling, and elementwise write in backend-native kernel code.
- Do not call PyTorch tanh, multiplication, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute `logits / soft_cap`, `tanh`, soft-cap multiplication, and final scale multiplication in FP32.
- Apply operations in the reference order: soft cap first, then scale.
- Cast only the final logits back to the configured dtype.
