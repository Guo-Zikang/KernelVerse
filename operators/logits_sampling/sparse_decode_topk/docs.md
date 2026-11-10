# sparse_decode_topk

## Function

Selects decode-time sparse attention column indices by applying top-k to each row's valid prefix.

## Inputs

- `logits`: tensor with shape `[num_rows, num_cols]`.
- `seq_lens`: int32 tensor with shape `[num_rows]`; row `i` only considers `logits[i, :seq_lens[i]]`.

All logits use the dtype selected by the shape entry.

## Output

- `indices`: int32 tensor with shape `[num_rows, topk]`.
- Rows with fewer than `topk` valid entries are padded with `-1`.

## Implementation Notes

- This is a sparse attention decode index selection helper, not attention score/value aggregation.
- Only the valid prefix from each row participates in top-k selection.
- Implement prefix-limited top-k and padding in backend-native kernel code.
- Break ties deterministically by smaller column index first.
- Do not call `torch.topk`, `torch.sort`, `torch.argsort`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 logits, compare scores in FP32 before top-k selection.
- Keep selected column indices and `-1` padding exact in INT32.
- Do not let low-precision rounding change selected indices.
