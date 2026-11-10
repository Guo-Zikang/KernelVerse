# sparse_prefill_topk

## Function

Selects prefill-time sparse attention column indices by applying top-k inside each row's candidate key range.

## Inputs

- `logits`: tensor with shape `[num_rows, num_cols]`.
- `cu_seqlen_ks`: int32 tensor with shape `[num_rows]`; row `i` starts at `cu_seqlen_ks[i]`.
- `cu_seqlen_ke`: int32 tensor with shape `[num_rows]`; row `i` ends at `cu_seqlen_ke[i]`.

All logits use the dtype selected by the shape entry.

## Output

- `indices`: int32 tensor with shape `[num_rows, topk]`.
- Indices are absolute column indices in `logits`.
- Rows with fewer than `topk` valid entries are padded with `-1`.

## Implementation Notes

- This is a sparse attention prefill index selection helper, not attention score/value aggregation.
- Only the range `[cu_seqlen_ks[row], cu_seqlen_ke[row])` participates in top-k selection.
- Implement range-limited top-k and padding in backend-native kernel code.
- Break ties deterministically by smaller absolute column index first.
- Do not call `torch.topk`, `torch.sort`, `torch.argsort`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 logits, compare scores in FP32 before top-k selection.
- Keep selected absolute column indices and `-1` padding exact in INT32.
- Do not let low-precision rounding change selected indices.
