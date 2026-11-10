"""Shape templates for cumsum benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dim": -1,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # last-dim scan, common hidden dimension pattern
        {"shape": [1, 4096], "dim": -1, "dtype": "FP32"},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32"},
        {"shape": [2048, 4096], "dim": -1, "dtype": "FP32"},
        {"shape": [4096, 2048], "dim": -1, "dtype": "FP32"},

        # first-axis scan, less contiguous memory access pattern
        {"shape": [512, 1024], "dim": 0, "dtype": "FP32"},
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP32"},
        {"shape": [8192, 1024], "dim": 0, "dtype": "FP32"},

        # sequence-axis scan for 3D tensors
        {"shape": [2, 128, 1024], "dim": 1, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "FP32"},
        {"shape": [8, 1024, 4096], "dim": 1, "dtype": "FP32"},

        # attention-like 4D layouts
        {"shape": [2, 8, 128, 64], "dim": 2, "dtype": "FP32"},
        {"shape": [4, 16, 512, 64], "dim": 2, "dtype": "FP32"},
        {"shape": [8, 32, 1024, 128], "dim": 3, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # last-dim hidden scan
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32"},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP16"},
        {"shape": [128, 4096], "dim": -1, "dtype": "BF16"},

        # sequence-axis scan for batched sequence tensors
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "FP16"},
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "BF16"},

        # first-axis scan
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP32"},
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP16"},
        {"shape": [4096, 2048], "dim": 0, "dtype": "BF16"},

        # attention-like 4D scan
        {"shape": [2, 8, 128, 64], "dim": 2, "dtype": "FP32"},
        {"shape": [2, 8, 128, 64], "dim": 2, "dtype": "FP16"},
        {"shape": [2, 8, 128, 64], "dim": 2, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [512, 512],
        [1024, 1024],
        [2048, 512],
        [256, 2048],
        [256, 1024],
        [1, 2048],
        [16, 2048],
        [128, 1024],
        [8, 128, 64],
        [16, 128, 64],
        [32, 512, 128],
        [64, 1024, 128],
        [2, 8, 128, 64],
        [4, 16, 512, 64],
        [8, 32, 1024, 128],
        [2, 4, 1024, 128],
        [4, 8, 256, 64],
    ],
    "dim": [-1, 0, 1, 2, 3],
    "dtype": ["FP32", "FP16", "BF16"],
}
