"""Shape templates for layer normalization benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 4096],
        "normalized_shape": [4096],
        "eps": 1e-5,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # hidden size sweep
        {"shape": [1, 1024], "normalized_shape": [1024], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [128, 1024], "normalized_shape": [1024], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [512, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [2048, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [512, 8192], "normalized_shape": [8192], "eps": 1e-5, "dtype": "FP32"},

        # batched sequence
        {"shape": [2, 128, 1024], "normalized_shape": [1024], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [8, 1024, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP32"},

        # multi-axis normalization
        {"shape": [16, 64, 32], "normalized_shape": [32], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [16, 64, 32], "normalized_shape": [64, 32], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [8, 128, 64], "normalized_shape": [64], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [8, 128, 64], "normalized_shape": [128, 64], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [4, 256, 128], "normalized_shape": [128], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [4, 256, 128], "normalized_shape": [256, 128], "eps": 1e-5, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # hidden vector
        {"shape": [128, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [128, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP16"},
        {"shape": [128, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "BF16"},

        # batched sequence
        {"shape": [4, 512, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "FP16"},
        {"shape": [4, 512, 4096], "normalized_shape": [4096], "eps": 1e-5, "dtype": "BF16"},

        # multi-axis
        {"shape": [16, 64, 32], "normalized_shape": [64, 32], "eps": 1e-5, "dtype": "FP32"},
        {"shape": [16, 64, 32], "normalized_shape": [64, 32], "eps": 1e-5, "dtype": "FP16"},
        {"shape": [16, 64, 32], "normalized_shape": [64, 32], "eps": 1e-5, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 512],
        [1, 1024],
        [1, 4096],
        [1, 8192],
        [128, 512],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [2048, 4096],
        [512, 8192],
        [2, 128, 1024],
        [4, 512, 4096],
        [8, 1024, 4096],
        [16, 64, 32],
        [8, 128, 64],
        [4, 256, 128],
        [2, 512, 256],
        [32, 16, 64],
        [2, 16, 64, 32],
        [4, 32, 32, 64],
        [8, 8, 128, 64],
    ],
    "normalized_shape": [
        [512],
        [1024],
        [4096],
        [8192],
        [32],
        [64],
        [128],
        [256],
        [64, 32],
        [128, 64],
        [256, 128],
        [512, 256],
        [16, 64],
        [32, 64],
    ],
    "eps": [1e-5, 1e-6],
    "dtype": ["FP32", "FP16", "BF16"],
}
