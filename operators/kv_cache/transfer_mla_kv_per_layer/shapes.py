"""Shape templates for transfer_mla_kv_per_layer benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_pages": 1024,
        "num_items": 256,
        "item_elems": 576,
        "dtype": "FP16",
        "index_dtype": "INT32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # copied page count sweep
        {"num_pages": 512, "num_items": 32, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 2048, "num_items": 256, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 8192, "num_items": 1024, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},

        # compressed MLA state width sweep
        {"num_pages": 2048, "num_items": 256, "item_elems": 512, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 2048, "num_items": 256, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 2048, "num_items": 256, "item_elems": 640, "dtype": "FP32", "index_dtype": "INT32"},

        # sparse versus dense relocation
        {"num_pages": 4096, "num_items": 1, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 4096, "num_items": 512, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 4096, "num_items": 4096, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
    ],
    "dtype_sweep": [
        # medium MLA cache transfer
        {"num_pages": 2048, "num_items": 256, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 2048, "num_items": 256, "item_elems": 576, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_pages": 2048, "num_items": 256, "item_elems": 576, "dtype": "BF16", "index_dtype": "INT32"},

        # wider compressed state transfer
        {"num_pages": 1024, "num_items": 128, "item_elems": 640, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 1024, "num_items": 128, "item_elems": 640, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_pages": 1024, "num_items": 128, "item_elems": 640, "dtype": "BF16", "index_dtype": "INT32"},

        # index dtype
        {"num_pages": 4096, "num_items": 512, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_pages": 4096, "num_items": 512, "item_elems": 576, "dtype": "FP32", "index_dtype": "INT64"},
        {"num_pages": 4096, "num_items": 512, "item_elems": 576, "dtype": "BF16", "index_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_pages": [16, 64, 256, 512, 1024, 2048, 4096, 8192],
    "num_items": [1, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096],
    "item_elems": [128, 256, 512, 576, 640, 768],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
}
