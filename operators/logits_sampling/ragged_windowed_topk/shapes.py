"""Shape templates for ragged windowed top-k benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch": 2,
        "length": 4096,
        "window_length": 4096,
        "topk": 2048,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # full-row windows
        {"batch": 2, "length": 4096, "window_length": 4096, "topk": 2048, "dtype": "FP32"},
        {"batch": 8, "length": 4096, "window_length": 4096, "topk": 1024, "dtype": "FP32"},
        {"batch": 32, "length": 4096, "window_length": 4096, "topk": 256, "dtype": "FP32"},

        # shifted partial windows
        {"batch": 32, "length": 4096, "window_length": 1024, "topk": 128, "dtype": "FP32"},
        {"batch": 128, "length": 8192, "window_length": 2048, "topk": 256, "dtype": "FP32"},
        {"batch": 512, "length": 32000, "window_length": 4096, "topk": 512, "dtype": "FP32"},

        # top-k ratio sweep
        {"batch": 128, "length": 4096, "window_length": 2048, "topk": 16, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "window_length": 2048, "topk": 256, "dtype": "FP32"},
        {"batch": 128, "length": 4096, "window_length": 2048, "topk": 1024, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # full-row window
        {"batch": 8, "length": 4096, "window_length": 4096, "topk": 1024, "dtype": "FP32"},
        {"batch": 8, "length": 4096, "window_length": 4096, "topk": 1024, "dtype": "FP16"},
        {"batch": 8, "length": 4096, "window_length": 4096, "topk": 1024, "dtype": "BF16"},

        # shifted partial window
        {"batch": 128, "length": 8192, "window_length": 2048, "topk": 256, "dtype": "FP32"},
        {"batch": 128, "length": 8192, "window_length": 2048, "topk": 256, "dtype": "FP16"},
        {"batch": 128, "length": 8192, "window_length": 2048, "topk": 256, "dtype": "BF16"},

        # large-row window
        {"batch": 512, "length": 32000, "window_length": 4096, "topk": 512, "dtype": "FP32"},
        {"batch": 512, "length": 32000, "window_length": 4096, "topk": 512, "dtype": "FP16"},
        {"batch": 512, "length": 32000, "window_length": 4096, "topk": 512, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch": [1, 2, 8, 32, 128, 512],
    "length": [256, 1024, 4096, 8192, 32000],
    "window_length": [64, 256, 1024, 2048, 4096],
    "topk": [1, 8, 16, 64, 128, 256, 512, 1024, 2048],
    "dtype": ["FP32", "FP16", "BF16"],
}
