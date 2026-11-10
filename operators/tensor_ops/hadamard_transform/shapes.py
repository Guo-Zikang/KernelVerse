"""Shape templates for hadamard_transform benchmark cases."""

SINGLE_SHAPES = {
    "default": {"shape": [128, 1024], "normalize": True, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # last-dimension sweep
        {"shape": [128, 128], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 256], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 512], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 1024], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 2048], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 4096], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 8192], "normalize": True, "dtype": "FP32"},

        # batch/token sweep
        {"shape": [1, 4096], "normalize": True, "dtype": "FP32"},
        {"shape": [32, 4096], "normalize": True, "dtype": "FP32"},
        {"shape": [512, 4096], "normalize": True, "dtype": "FP32"},

        # batched sequence layouts
        {"shape": [2, 128, 1024], "normalize": True, "dtype": "FP32"},
        {"shape": [4, 512, 2048], "normalize": True, "dtype": "FP32"},
        {"shape": [8, 1024, 4096], "normalize": True, "dtype": "FP32"},

        # normalization flag sweep
        {"shape": [128, 1024], "normalize": False, "dtype": "FP32"},
        {"shape": [512, 2048], "normalize": False, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "normalize": False, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # hidden vector
        {"shape": [128, 1024], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 1024], "normalize": True, "dtype": "FP16"},
        {"shape": [128, 1024], "normalize": True, "dtype": "BF16"},

        # larger hidden vector
        {"shape": [128, 4096], "normalize": True, "dtype": "FP32"},
        {"shape": [128, 4096], "normalize": True, "dtype": "FP16"},
        {"shape": [128, 4096], "normalize": True, "dtype": "BF16"},

        # batched sequence layout
        {"shape": [4, 512, 2048], "normalize": True, "dtype": "FP32"},
        {"shape": [4, 512, 2048], "normalize": True, "dtype": "FP16"},
        {"shape": [4, 512, 2048], "normalize": True, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 128],
        [1, 512],
        [1, 1024],
        [1, 4096],
        [32, 512],
        [64, 1024],
        [128, 1024],
        [128, 2048],
        [128, 4096],
        [512, 4096],
        [2, 128, 1024],
        [4, 512, 2048],
        [8, 1024, 4096],
        [2, 16, 64, 512],
        [4, 32, 32, 1024],
        [8, 8, 128, 2048],
        [16, 1, 8192],
        [256, 8192],
        [1024, 1024],
        [2, 2048, 1024],
        [4, 16, 256, 4096],
    ],
    "normalize": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
