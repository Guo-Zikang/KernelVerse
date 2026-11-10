"""Shape templates for transfer_kv_all_layer benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_layers": 32,
        "num_pages": 1024,
        "num_items": 256,
        "item_elems": 128,
        "dtype": "FP16",
        "index_dtype": "INT32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # layer-count sweep
        {"num_layers": 1, "num_pages": 1024, "num_items": 256, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 16, "num_pages": 1024, "num_items": 256, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 48, "num_pages": 1024, "num_items": 256, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},

        # copied page count sweep
        {"num_layers": 24, "num_pages": 512, "num_items": 32, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 24, "num_pages": 2048, "num_items": 256, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 24, "num_pages": 8192, "num_items": 1024, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},

        # item size sweep
        {"num_layers": 24, "num_pages": 2048, "num_items": 256, "item_elems": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 24, "num_pages": 2048, "num_items": 256, "item_elems": 256, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 24, "num_pages": 2048, "num_items": 256, "item_elems": 512, "dtype": "FP32", "index_dtype": "INT32"},
    ],
    "dtype_sweep": [
        # medium all-layer transfer
        {"num_layers": 32, "num_pages": 1024, "num_items": 256, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 32, "num_pages": 1024, "num_items": 256, "item_elems": 128, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_layers": 32, "num_pages": 1024, "num_items": 256, "item_elems": 128, "dtype": "BF16", "index_dtype": "INT32"},

        # wider page row transfer
        {"num_layers": 24, "num_pages": 1024, "num_items": 128, "item_elems": 512, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 24, "num_pages": 1024, "num_items": 128, "item_elems": 512, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_layers": 24, "num_pages": 1024, "num_items": 128, "item_elems": 512, "dtype": "BF16", "index_dtype": "INT32"},

        # index dtype
        {"num_layers": 48, "num_pages": 4096, "num_items": 512, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_layers": 48, "num_pages": 4096, "num_items": 512, "item_elems": 128, "dtype": "FP32", "index_dtype": "INT64"},
        {"num_layers": 48, "num_pages": 4096, "num_items": 512, "item_elems": 128, "dtype": "BF16", "index_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_layers": [1, 2, 8, 16, 24, 32, 48],
    "num_pages": [16, 64, 256, 512, 1024, 2048, 4096, 8192],
    "num_items": [1, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096],
    "item_elems": [16, 32, 64, 80, 96, 128, 256, 512],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
}
