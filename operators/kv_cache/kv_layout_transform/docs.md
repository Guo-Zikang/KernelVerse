# kv_layout_transform

## Function

Transforms K/V cache data between paged layouts and layer-first flat layouts.

## Inputs

- `src_k`: source key cache tensor. Shape depends on `layout_mode`.
- `dst_k`: destination key cache tensor. Shape depends on `layout_mode`.
- `src_v`: source value cache tensor. Shape matches `src_k`.
- `dst_v`: destination value cache tensor. Shape matches `dst_k`.
- `src_indices`: source token indices with shape `[num_items]`.
- `dst_indices`: destination token indices with shape `[num_items]`.

Floating tensors use the dtype selected by the shape entry. Indices use `index_dtype`.

## Output

Returns a tuple:

- `out_k`: transformed key cache tensor with the same shape as `dst_k`.
- `out_v`: transformed value cache tensor with the same shape as `dst_v`.

## Layout Modes

- `per_layer_pf_lf`: page-first `[num_pages, num_layers, page_size, item_elems]` to one layer-first flat output `[num_pages * page_size, item_elems]`.
- `per_layer_ph_lf`: page-head `[num_pages, head_num, page_size, num_layers, head_elems]` to one layer-first flat output `[num_pages * page_size, head_num, head_elems]`.
- `all_layer_lf_pf`: all-layer layer-first `[num_layers, num_pages * page_size, item_elems]` to page-first `[num_pages, num_layers, page_size, item_elems]`.
- `all_layer_lf_ph`: all-layer layer-first `[num_layers, num_pages * page_size, head_num, head_elems]` to page-head `[num_pages, head_num, page_size, num_layers, head_elems]`.

## Implementation Notes

- This is a memory-bound KV cache layout transform operator.
- Preserve destination slots that are not listed in `dst_indices`.
- Use `src_idx // page_size` and `src_idx % page_size` or the corresponding destination decomposition when mapping flat token indices to paged coordinates.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a layout transform operator; only change memory layout, not numerical values.
- Preserve K/V cache values exactly in the configured dtype.
- Keep flat-to-paged index decomposition exact and preserve destination slots that are not listed.
