"""Shape templates for var_mean benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dim": -1,
        "dtype": "FP32",
        "keepdim": False,
        "correction": 1,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # last-dim reduction, common hidden reduction
        {"shape": [1, 4096], "dim": -1, "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "keepdim": False, "correction": 0},
        {"shape": [2048, 4096], "dim": -1, "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [2048, 4096], "dim": -1, "dtype": "FP32", "keepdim": True, "correction": 0},

        # first-axis reduction, less contiguous memory access pattern
        {"shape": [512, 1024], "dim": 0, "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP32", "keepdim": False, "correction": 0},
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP32", "keepdim": True, "correction": 1},

        # middle-axis reduction for 3D tensors
        {"shape": [2, 128, 1024], "dim": 1, "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [8, 1024, 4096], "dim": 1, "dtype": "FP32", "keepdim": False, "correction": 0},
        {"shape": [8, 1024, 4096], "dim": 1, "dtype": "FP32", "keepdim": True, "correction": 1},

        # attention-like 4D layouts
        {"shape": [2, 8, 128, 64], "dim": 2, "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [4, 16, 512, 64], "dim": 2, "dtype": "FP32", "keepdim": True, "correction": 0},
        {"shape": [8, 32, 1024, 128], "dim": 3, "dtype": "FP32", "keepdim": False, "correction": 1},

        # multi-axis reductions
        {"shape": [2, 8, 128, 64], "dim": [2, 3], "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [4, 16, 512, 64], "dim": [1, 2], "dtype": "FP32", "keepdim": True, "correction": 0},
        {"shape": [8, 32, 1024, 128], "dim": [-2, -1], "dtype": "FP32", "keepdim": False, "correction": 1},
    ],
    "dtype_sweep": [
        # last-dim hidden reduction
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "keepdim": False, "correction": 1},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP16", "keepdim": False, "correction": 1},
        {"shape": [128, 4096], "dim": -1, "dtype": "BF16", "keepdim": False, "correction": 1},

        # middle-axis reduction for batched sequence tensors
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "FP32", "keepdim": False, "correction": 0},
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "FP16", "keepdim": False, "correction": 0},
        {"shape": [4, 512, 4096], "dim": 1, "dtype": "BF16", "keepdim": False, "correction": 0},

        # first-axis reduction with keepdim enabled
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP32", "keepdim": True, "correction": 1},
        {"shape": [4096, 2048], "dim": 0, "dtype": "FP16", "keepdim": True, "correction": 1},
        {"shape": [4096, 2048], "dim": 0, "dtype": "BF16", "keepdim": True, "correction": 1},

        # multi-axis reduction for attention-like 4D tensors
        {"shape": [2, 8, 128, 64], "dim": [2, 3], "dtype": "FP32", "keepdim": False, "correction": 0},
        {"shape": [2, 8, 128, 64], "dim": [2, 3], "dtype": "FP16", "keepdim": False, "correction": 0},
        {"shape": [2, 8, 128, 64], "dim": [2, 3], "dtype": "BF16", "keepdim": False, "correction": 0},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [512, 512],
        [1024, 1024],
        [2048, 4096],
        [4096, 2048],
        [8192, 1024],
        [1, 4096],
        [16, 2048],
        [128, 4096],
        [8, 128, 64],
        [16, 128, 64],
        [32, 512, 128],
        [64, 1024, 128],
        [2, 8, 128, 64],
        [4, 16, 512, 64],
        [8, 32, 1024, 128],
        [2, 4, 1024, 128],
        [4, 8, 2048, 64],
    ],
    "dim": [-1, 0, 1, 2, [0, 1], [-2, -1], [1, 2], [0, 2]],
    "dtype": ["FP32", "FP16", "BF16"],
    "keepdim": [True, False],
    "correction": [0, 1],
}
