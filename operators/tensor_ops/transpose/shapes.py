"""Shape templates for transpose benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 32, 128],
        "dim0": 0,
        "dim1": 1,
        "dtype": "BF16",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # matrix transpose and identity-like swaps for 2D tensors
        {"shape": [1024, 1024], "dim0": 0, "dim1": 1, "dtype": "FP32"},
        {"shape": [2048, 4096], "dim0": 0, "dim1": 1, "dtype": "FP32"},
        {"shape": [4096, 2048], "dim0": -2, "dim1": -1, "dtype": "FP32"},

        # sequence and hidden dimension swaps for 3D tensors
        {"shape": [16, 128, 64], "dim0": 1, "dim1": 2, "dtype": "FP32"},
        {"shape": [32, 512, 128], "dim0": 0, "dim1": 1, "dtype": "FP32"},
        {"shape": [64, 1024, 128], "dim0": -2, "dim1": -1, "dtype": "FP32"},
        {"shape": [128, 8, 128], "dim0": 0, "dim1": 1, "dtype": "FP32"},

        # attention-like 4D and 5D layouts
        {"shape": [2, 128, 8, 64], "dim0": 1, "dim1": 2, "dtype": "FP32"},
        {"shape": [4, 512, 16, 64], "dim0": 1, "dim1": 2, "dtype": "FP32"},
        {"shape": [8, 1024, 32, 128], "dim0": 1, "dim1": 2, "dtype": "FP32"},
        {"shape": [2, 8, 16, 64, 128], "dim0": 1, "dim1": 3, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # 2D matrix transpose
        {"shape": [1024, 1024], "dim0": 0, "dim1": 1, "dtype": "FP32"},
        {"shape": [1024, 1024], "dim0": 0, "dim1": 1, "dtype": "FP16"},
        {"shape": [1024, 1024], "dim0": 0, "dim1": 1, "dtype": "BF16"},

        # 3D batch/sequence swap
        {"shape": [128, 32, 128], "dim0": 0, "dim1": 1, "dtype": "FP32"},
        {"shape": [128, 32, 128], "dim0": 0, "dim1": 1, "dtype": "FP16"},
        {"shape": [128, 32, 128], "dim0": 0, "dim1": 1, "dtype": "BF16"},

        # integer transpose cases
        {"shape": [128, 8, 128], "dim0": 0, "dim1": 1, "dtype": "INT64"},
        {"shape": [32, 128, 128], "dim0": 0, "dim1": 1, "dtype": "INT32"},
        {"shape": [2, 128, 8, 64], "dim0": 1, "dim1": 2, "dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [512, 512],
        [1024, 1024],
        [2048, 4096],
        [4096, 2048],
        [8192, 1024],
        [2048, 11008],
        [8, 128, 64],
        [16, 128, 64],
        [32, 512, 128],
        [64, 1024, 128],
        [128, 8, 128],
        [2, 128, 8, 64],
        [2, 8, 128, 64],
        [4, 512, 16, 64],
        [8, 1024, 32, 128],
        [2, 4, 16, 128, 64],
        [16384, 256],
        [4, 64, 256],
        [8, 256, 512],
        [16, 64, 1024],
        [1, 2048, 8, 128],
        [16, 256, 8, 64],
        [2, 32, 32, 128],
        [4, 8, 512, 64],
        [2, 8, 16, 64, 128],
    ],
    "dim0": [0, 1, 2, -2],
    "dim1": [1, 2, 3, -1],
    "dtype": ["FP32", "FP16", "BF16", "INT32", "INT64"],
}
