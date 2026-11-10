# mla_kv_layout_transform

## Function

Transforms compressed MLA KV-cache data between paged storage and layer-first flat storage.

## Inputs

- `src`: source compressed MLA cache tensor. Shape depends on `layout_mode`.
- `dst`: destination compressed MLA cache tensor. Shape depends on `layout_mode`.
- `src_indices`: source token indices with shape `[num_items]`.
- `dst_indices`: destination token indices with shape `[num_items]`.

Floating tensors use the dtype selected by the shape entry. Indices use `index_dtype`.

## Output

Returns one transformed compressed MLA cache tensor with the same shape as `dst`.

## Layout Modes

- `per_layer_pf_lf`: page-first `[num_pages, num_layers, page_size, item_elems]` to one layer-first flat output `[num_pages * page_size, item_elems]`.
- `all_layer_lf_pf`: all-layer layer-first `[num_layers, num_pages * page_size, item_elems]` to page-first `[num_pages, num_layers, page_size, item_elems]`.

## Implementation Notes

- This is a memory-bound MLA KV-cache layout transform operator.
- Preserve destination slots that are not listed in `dst_indices`.
- Use `src_idx // page_size` and `src_idx % page_size` or the corresponding destination decomposition when mapping flat token indices to paged coordinates.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a layout transform operator; only change memory layout, not numerical values.
- Preserve compressed MLA cache values exactly in the configured dtype.
- Keep flat-to-paged index decomposition exact and preserve destination slots that are not listed.
