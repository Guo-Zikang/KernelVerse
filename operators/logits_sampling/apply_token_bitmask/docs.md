# apply_token_bitmask

## Function

Masks disallowed token logits using a packed token bitmask before sampling.

## Inputs

- `logits`: tensor with shape `[batch, vocab_size]`.
- `bitmask`: packed `INT32` bitmask with shape `[batch, vocab_size // 32]`.
- `indices`: optional `INT32` row indices. If empty, all rows are processed.

`logits` uses the dtype selected by the shape entry.

## Output

Returns masked logits with shape `[batch, vocab_size]` and the same dtype as `logits`.

## Bitmask Semantics

- `vocab_size` must be divisible by 32.
- Each `INT32` bitmask word stores allowed/disallowed flags for 32 token ids.
- A bit value of 1 means the token is allowed.
- A bit value of 0 means the token logit is set to negative infinity.

## Implementation Notes

- Implement packed bit decoding and logits masking in backend-native kernel code.
- If `indices` is empty, process all rows.
- If `indices` is non-empty, only process the selected rows.
- Do not call PyTorch indexing, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Preserve unmasked logits exactly; do not rescale or normalize them.
- Write masked logits as negative infinity in the configured output dtype.
- Keep bit decoding and row index handling exact in integer logic.
