# chunk_local_cumsum

## Function

Computes cumulative sums independently inside each fixed-size chunk along the sequence dimension.

When `reverse` is true, the cumulative sum runs from the end of each chunk toward the beginning.

## Inputs

- `g`: gate tensor with shape `[batch_size, seq_len, num_heads]` or `[batch_size, seq_len, num_heads, width]`.

The input tensor uses the dtype selected by the shape entry.

## Output

Returns one tensor with the same shape as `g`.

The output uses the dtype selected by the shape entry.

## Implementation Notes

- Implement the chunk-local forward or reverse cumulative sum in backend-native kernel code.
- For FP16/BF16 inputs, keep the chunk-local running sum in FP32 for both forward and reverse scans.
- Do not store scan carry values in FP16/BF16 between positions; cast to the output dtype only when writing final outputs.
- Do not call `torch.cumsum`, `torch.flip`, or other PyTorch operators for the core computation in `ModelNew.forward`.
