"""Shape templates for variable-length top-k benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch": 128,
        "length": 4096,
        "topk": 16,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # batch-size sweep
        {"batch": 1, "length": 4096, "topk": 16, "dtype": "FP32"},
        {"batch": 32, "length": 4096, "topk": 16, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "topk": 16, "dtype": "FP32"},
        {"batch": 512, "length": 4096, "topk": 16, "dtype": "FP32"},

        # length sweep
        {"batch": 128, "length": 256, "topk": 8, "dtype": "FP32"},
        {"batch": 128, "length": 1024, "topk": 16, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "topk": 32, "dtype": "FP32"},
        {"batch": 64, "length": 32000, "topk": 64, "dtype": "FP32"},

        # top-k sweep
        {"batch": 128, "length": 4096, "topk": 1, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "topk": 8, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "topk": 64, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # score dtype sweep
        {"batch": 128, "length": 4096, "topk": 16, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "topk": 16, "dtype": "FP16"},
        {"batch": 128, "length": 4096, "topk": 16, "dtype": "BF16"},

        # large vocabulary-like length
        {"batch": 64, "length": 32000, "topk": 64, "dtype": "FP32"},
        {"batch": 64, "length": 32000, "topk": 64, "dtype": "FP16"},
        {"batch": 64, "length": 32000, "topk": 64, "dtype": "BF16"},

        # short length
        {"batch": 512, "length": 1024, "topk": 8, "dtype": "FP32"},
        {"batch": 512, "length": 1024, "topk": 8, "dtype": "FP16"},
        {"batch": 512, "length": 1024, "topk": 8, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch": [1, 8, 32, 64, 128, 256, 512, 1024],
    "length": [8, 16, 64, 256, 1024, 4096, 8192, 32000],
    "topk": [1, 2, 4, 8, 16, 32, 64, 100],
    "dtype": ["FP32", "FP16", "BF16"],
}
