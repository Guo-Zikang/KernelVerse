# transfer_mla_kv_per_layer

## Function

Copies selected compressed MLA KV-cache rows for one transformer layer from source cache slots to destination cache slots.

## Inputs

- `src`: source compressed MLA cache tensor with shape `[num_pages, item_elems]`.
- `dst`: destination compressed MLA cache tensor with shape `[num_pages, item_elems]`.
- `src_indices`: source row indices with shape `[num_items]`.
- `dst_indices`: destination row indices with shape `[num_items]`.

`src` and `dst` use the dtype selected by the shape entry. Indices use `index_dtype`.

## Output

Returns one copied cache tensor with shape `[num_pages, item_elems]`.

## Implementation Notes

- This is a memory-bound MLA KV-cache relocation operator.
- For each `i`, copy `src[src_indices[i]]` to `out[dst_indices[i]]`.
- Preserve destination rows that are not listed in `dst_indices`.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache relocation operator; copy source rows to destination rows without changing values.
- Preserve source compressed MLA cache values exactly in the configured dtype.
- Keep source/destination index handling exact and preserve destination rows that are not listed.
