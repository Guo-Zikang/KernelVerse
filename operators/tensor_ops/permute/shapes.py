"""Shape templates for permute benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 32, 128],
        "perm": [1, 0, 2],
        "dtype": "BF16",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # matrix transpose and identity paths for 2D tensors
        {"shape": [1024, 1024], "perm": [1, 0], "dtype": "FP32"},
        {"shape": [2048, 4096], "perm": [1, 0], "dtype": "FP32"},
        {"shape": [4096, 2048], "perm": [0, 1], "dtype": "FP32"},

        # sequence and hidden dimension permutations for 3D tensors
        {"shape": [16, 128, 64], "perm": [0, 2, 1], "dtype": "FP32"},
        {"shape": [32, 512, 128], "perm": [1, 0, 2], "dtype": "FP32"},
        {"shape": [64, 1024, 128], "perm": [2, 0, 1], "dtype": "FP32"},
        {"shape": [128, 8, 128], "perm": [0, 1, 2], "dtype": "FP32"},

        # attention-like 4D layouts
        {"shape": [2, 128, 8, 64], "perm": [0, 2, 1, 3], "dtype": "FP32"},
        {"shape": [4, 512, 16, 64], "perm": [0, 2, 1, 3], "dtype": "FP32"},
        {"shape": [8, 1024, 32, 128], "perm": [0, 2, 1, 3], "dtype": "FP32"},
        {"shape": [2, 8, 16, 64, 128], "perm": [0, 2, 1, 3, 4], "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # 2D matrix transpose
        {"shape": [1024, 1024], "perm": [1, 0], "dtype": "FP32"},
        {"shape": [1024, 1024], "perm": [1, 0], "dtype": "FP16"},
        {"shape": [1024, 1024], "perm": [1, 0], "dtype": "BF16"},

        # 3D batch/sequence swap
        {"shape": [128, 32, 128], "perm": [1, 0, 2], "dtype": "FP32"},
        {"shape": [128, 32, 128], "perm": [1, 0, 2], "dtype": "FP16"},
        {"shape": [128, 32, 128], "perm": [1, 0, 2], "dtype": "BF16"},

        # integer permutation cases
        {"shape": [128, 8, 128], "perm": [0, 1, 2], "dtype": "INT64"},
        {"shape": [32, 128, 128], "perm": [1, 0, 2], "dtype": "INT32"},
        {"shape": [2, 128, 8, 64], "perm": [0, 2, 1, 3], "dtype": "INT64"},
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
    "perm": [
        [1, 0],
        [0, 1],
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
        [2, 0, 1],
        [0, 2, 1, 3],
        [0, 1, 3, 2],
        [0, 3, 1, 2],
        [1, 0, 2, 3],
        [2, 0, 1, 3],
        [3, 0, 1, 2],
        [0, 1, 2, 4, 3],
        [0, 2, 1, 3, 4],
        [0, 3, 1, 2, 4],
    ],
    "dtype": ["FP32", "FP16", "BF16", "INT32", "INT64"],
}
