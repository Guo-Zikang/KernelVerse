# paged_windowed_topk

## Function

Selects top-k score positions from each row within a valid window, then maps the selected positions through a page table.

## Inputs

- `score`: input score tensor with shape `[batch, length]`.
- `lengths`: int32 tensor with shape `[batch]`.
- `page_table`: int32 tensor with shape `[batch, length]`.
- `row_starts`: int32 tensor with shape `[batch]`.

All floating input tensors use the dtype selected by the shape entry.
Each row only considers `score[row, row_starts[row]:row_starts[row] + lengths[row]]` as valid.

## Output

- `indices`: int32 tensor with shape `[batch, topk]`.

## Implementation Notes

- Select local top-k positions inside each valid window.
- Add `row_starts[row]` to convert local top-k positions into absolute row positions.
- Return `page_table[row, absolute_position]`, not the absolute row position itself.
- Only row-wise top-k over contiguous windows is required for this operator.
- The generated `lengths[row]` values are at least `topk`.
- Break ties deterministically by smaller local window index first before mapping through `page_table`.
- Do not call `torch.topk`, `torch.sort`, `torch.argsort`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compare scores in FP32 before top-k selection.
- Keep mapped index outputs exact in INT32 and preserve deterministic tie-breaking.
- Do not let low-precision rounding change selected positions before page-table lookup.
