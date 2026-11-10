"""Shape templates for GELU activation benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # vector and hidden-size sweep
        {"shape": [1, 1024], "dtype": "FP32"},
        {"shape": [1, 4096], "dtype": "FP32"},
        {"shape": [128, 1024], "dtype": "FP32"},
        {"shape": [128, 4096], "dtype": "FP32"},
        {"shape": [512, 8192], "dtype": "FP32"},

        # batched sequence layouts
        {"shape": [2, 128, 1024], "dtype": "FP32"},
        {"shape": [4, 512, 4096], "dtype": "FP32"},
        {"shape": [8, 1024, 4096], "dtype": "FP32"},

        # 4D activation layouts
        {"shape": [2, 16, 64, 32], "dtype": "FP32"},
        {"shape": [4, 32, 32, 64], "dtype": "FP32"},
        {"shape": [8, 8, 128, 64], "dtype": "FP32"},
    ],
    "dtype_sweep": [
        {"shape": [128, 4096], "dtype": "FP32"},
        {"shape": [128, 4096], "dtype": "FP16"},
        {"shape": [128, 4096], "dtype": "BF16"},

        {"shape": [4, 512, 4096], "dtype": "FP32"},
        {"shape": [4, 512, 4096], "dtype": "FP16"},
        {"shape": [4, 512, 4096], "dtype": "BF16"},

        {"shape": [4, 32, 32, 64], "dtype": "FP32"},
        {"shape": [4, 32, 32, 64], "dtype": "FP16"},
        {"shape": [4, 32, 32, 64], "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 2048],
        [1, 4096],
        [1, 8192],
        [1, 16384],
        [8, 2048],
        [16, 2048],
        [32, 4096],
        [64, 4096],
        [128, 2048],
        [128, 4096],
        [128, 8192],
        [256, 4096],
        [256, 8192],
        [512, 8192],
        [512, 16384],
        [1024, 8192],
        [2, 128, 2048],
        [2, 512, 4096],
        [4, 512, 8192],
        [4, 1024, 8192],
        [8, 1024, 8192],
        [16, 256, 16384],
        [2, 256, 1024],
        [4, 1024, 4096],
        [8, 2048, 4096],
        [16, 512, 8192],
        [2, 16, 64, 2048],
        [4, 32, 32, 4096],
        [4, 32, 32, 8192],
        [8, 8, 128, 4096],
        [2, 8, 512, 128],
        [2, 16, 256, 256],
        [4, 16, 512, 512],
        [8, 16, 128, 1024],
        [16, 8, 256, 2048],
    ],
    "dtype": ["FP32", "FP16", "BF16"],
}
