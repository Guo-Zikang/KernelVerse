"""Shape templates for where benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dtype": "FP32",
        "condition_probability": 0.5,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # 2D hidden and projection-like masks
        {"shape": [1, 4096], "dtype": "FP32", "condition_probability": 0.25},
        {"shape": [128, 1024], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [128, 4096], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [512, 8192], "dtype": "FP32", "condition_probability": 0.75},

        # batched sequence masks
        {"shape": [2, 128, 1024], "dtype": "FP32", "condition_probability": 0.25},
        {"shape": [4, 512, 4096], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [8, 1024, 4096], "dtype": "FP32", "condition_probability": 0.75},

        # attention-like 4D layouts
        {"shape": [2, 8, 128, 64], "dtype": "FP32", "condition_probability": 0.25},
        {"shape": [4, 16, 512, 64], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [8, 32, 1024, 128], "dtype": "FP32", "condition_probability": 0.75},
        {"shape": [16, 16, 2048, 128], "dtype": "FP32", "condition_probability": 0.5},
    ],
    "dtype_sweep": [
        # 2D hidden selection
        {"shape": [128, 4096], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [128, 4096], "dtype": "FP16", "condition_probability": 0.5},
        {"shape": [128, 4096], "dtype": "BF16", "condition_probability": 0.5},

        # 3D sequence selection
        {"shape": [4, 512, 4096], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [4, 512, 4096], "dtype": "FP16", "condition_probability": 0.5},
        {"shape": [4, 512, 4096], "dtype": "BF16", "condition_probability": 0.5},

        # attention-like selection
        {"shape": [2, 16, 512, 64], "dtype": "FP32", "condition_probability": 0.5},
        {"shape": [2, 16, 512, 64], "dtype": "FP16", "condition_probability": 0.5},
        {"shape": [2, 16, 512, 64], "dtype": "BF16", "condition_probability": 0.5},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 4096],
        [128, 1024],
        [128, 4096],
        [256, 4096],
        [512, 8192],
        [1024, 8192],
        [2048, 8192],
        [4096, 4096],
        [2, 128, 1024],
        [2, 2048, 4096],
        [4, 512, 4096],
        [8, 1024, 4096],
        [16, 2048, 4096],
        [2, 8, 128, 64],
        [4, 16, 512, 64],
        [8, 32, 1024, 128],
        [16, 16, 2048, 128],
        [2, 4, 1024, 128],
        [4, 8, 2048, 64],
        [8, 16, 2048, 128],
    ],
    "dtype": ["FP32", "FP16", "BF16"],
    "condition_probability": [0.25, 0.5, 0.75],
}
