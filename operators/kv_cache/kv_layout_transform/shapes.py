"""Shape templates for kv_layout_transform benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "layout_mode": "per_layer_pf_lf",
        "num_layers": 32,
        "num_pages": 1024,
        "page_size": 16,
        "num_items": 512,
        "item_elems": 128,
        "layer_id": 1,
        "dtype": "FP16",
        "index_dtype": "INT32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # page-first to layer-first, single layer
        {"layout_mode": "per_layer_pf_lf", "num_layers": 32, "num_pages": 512, "page_size": 16, "num_items": 128, "item_elems": 128, "layer_id": 1, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_pf_lf", "num_layers": 32, "num_pages": 2048, "page_size": 16, "num_items": 512, "item_elems": 128, "layer_id": 8, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_pf_lf", "num_layers": 32, "num_pages": 2048, "page_size": 16, "num_items": 1024, "item_elems": 128, "layer_id": 16, "dtype": "FP32", "index_dtype": "INT32"},

        # page-head to layer-first, single layer
        {"layout_mode": "per_layer_ph_lf", "num_layers": 32, "num_pages": 512, "page_size": 16, "num_items": 128, "head_num": 8, "head_elems": 128, "layer_id": 1, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_ph_lf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "head_num": 8, "head_elems": 128, "layer_id": 8, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_ph_lf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 1024, "head_num": 8, "head_elems": 128, "layer_id": 16, "dtype": "FP32", "index_dtype": "INT32"},

        # layer-first to paged layouts, all layers
        {"layout_mode": "all_layer_lf_pf", "num_layers": 16, "num_pages": 1024, "page_size": 16, "num_items": 256, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "all_layer_lf_pf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "all_layer_lf_ph", "num_layers": 16, "num_pages": 512, "page_size": 16, "num_items": 512, "head_num": 8, "head_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
    ],
    "dtype_sweep": [
        # page-first to layer-first
        {"layout_mode": "per_layer_pf_lf", "num_layers": 32, "num_pages": 2048, "page_size": 16, "num_items": 512, "item_elems": 128, "layer_id": 8, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_pf_lf", "num_layers": 32, "num_pages": 2048, "page_size": 16, "num_items": 512, "item_elems": 128, "layer_id": 8, "dtype": "FP16", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_pf_lf", "num_layers": 32, "num_pages": 2048, "page_size": 16, "num_items": 512, "item_elems": 128, "layer_id": 8, "dtype": "BF16", "index_dtype": "INT32"},

        # page-head to layer-first
        {"layout_mode": "per_layer_ph_lf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "head_num": 8, "head_elems": 128, "layer_id": 8, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_ph_lf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "head_num": 8, "head_elems": 128, "layer_id": 8, "dtype": "FP16", "index_dtype": "INT32"},
        {"layout_mode": "per_layer_ph_lf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "head_num": 8, "head_elems": 128, "layer_id": 8, "dtype": "BF16", "index_dtype": "INT32"},

        # index dtype
        {"layout_mode": "all_layer_lf_pf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"layout_mode": "all_layer_lf_pf", "num_layers": 32, "num_pages": 1024, "page_size": 16, "num_items": 512, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT64"},
        {"layout_mode": "all_layer_lf_ph", "num_layers": 16, "num_pages": 512, "page_size": 16, "num_items": 512, "head_num": 8, "head_elems": 128, "dtype": "BF16", "index_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "layout_mode": ["per_layer_pf_lf", "per_layer_ph_lf", "all_layer_lf_pf", "all_layer_lf_ph"],
    "num_layers": [1, 2, 8, 16],
    "num_pages": [16, 64, 256, 512, 1024],
    "page_size": [1, 4, 8, 16],
    "num_items": [1, 8, 16, 32, 128, 256, 512, 1024],
    "item_elems": [64, 80, 96, 128],
    "head_num": [1, 4],
    "head_elems": [64, 80, 96, 128],
    "layer_id": [0, 1, 7, 15],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
}
