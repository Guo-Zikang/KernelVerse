"""Shape templates for repeat_interleave benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dim": 1,
        "repeats": 2,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # flatten repeat-interleave cases
        {"shape": [128, 1024], "dim": None, "repeats": 2, "dtype": "FP32"},
        {"shape": [2, 128, 1024], "dim": None, "repeats": 2, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "dim": None, "repeats": 2, "dtype": "FP32"},

        # batch and feature axis repeat-interleave for 2D tensors
        {"shape": [512, 1024], "dim": 0, "repeats": 2, "dtype": "FP32"},
        {"shape": [1024, 4096], "dim": 0, "repeats": 2, "dtype": "FP32"},
        {"shape": [2048, 4096], "dim": 1, "repeats": 4, "dtype": "FP32"},

        # sequence and hidden dimension repeat-interleave for 3D tensors
        {"shape": [2, 128, 1024], "dim": 1, "repeats": 2, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "dim": 1, "repeats": 2, "dtype": "FP32"},
        {"shape": [8, 1024, 4096], "dim": -1, "repeats": 2, "dtype": "FP32"},

        # attention-like 4D layouts
        {"shape": [2, 8, 128, 64], "dim": 2, "repeats": 2, "dtype": "FP32"},
        {"shape": [4, 16, 512, 64], "dim": 1, "repeats": 2, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # 2D feature repeat-interleave
        {"shape": [128, 1024], "dim": 1, "repeats": 2, "dtype": "FP32"},
        {"shape": [128, 1024], "dim": 1, "repeats": 2, "dtype": "FP16"},
        {"shape": [128, 1024], "dim": 1, "repeats": 2, "dtype": "BF16"},

        # 3D hidden repeat-interleave
        {"shape": [2, 128, 1024], "dim": -1, "repeats": 2, "dtype": "FP32"},
        {"shape": [2, 128, 1024], "dim": -1, "repeats": 2, "dtype": "FP16"},
        {"shape": [2, 128, 1024], "dim": -1, "repeats": 2, "dtype": "BF16"},

        # integer repeat-interleave cases
        {"shape": [128, 1024], "dim": 1, "repeats": 2, "dtype": "INT32"},
        {"shape": [2, 128, 1024], "dim": -1, "repeats": 2, "dtype": "INT64"},
        {"shape": [4, 512, 4096], "dim": None, "repeats": 2, "dtype": "INT32"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [512, 512],
        [1024, 1024],
        [2048, 4096],
        [4096, 2048],
        [8192, 1024],
        [1024, 8192],
        [2048, 8192],
        [1, 4096],
        [16, 2048],
        [128, 4096],
        [2, 128, 1024],
        [4, 512, 4096],
        [8, 1024, 4096],
        [2, 8, 128, 64],
        [4, 16, 512, 64],
        [8, 32, 1024, 128],
        [16, 16, 2048, 128],
        [2, 4, 1024, 128],
        [4, 8, 2048, 64],
    ],
    "dim": [None, -1, 0, 1, 2, 3],
    "repeats": [2, 3, 4],
    "dtype": ["FP32", "FP16", "BF16", "INT32", "INT64"],
}
