"""Shape templates for sort benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dim": -1,
        "dtype": "FP32",
        "descending": False,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # last-dim sort over vocab or hidden dimensions
        {"shape": [1, 32000], "dim": -1, "dtype": "FP32", "descending": True},
        {"shape": [8, 32000], "dim": -1, "dtype": "FP32", "descending": True},
        {"shape": [32, 32000], "dim": -1, "dtype": "FP32", "descending": True},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "descending": False},

        # feature-axis sort for 3D tensors
        {"shape": [64, 256, 1024], "dim": 1, "dtype": "FP32", "descending": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP32", "descending": False},
        {"shape": [16, 1024, 4096], "dim": 1, "dtype": "FP32", "descending": False},

        # attention score layouts
        {"shape": [2, 8, 128, 128], "dim": -1, "dtype": "FP32", "descending": True},
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "FP32", "descending": True},
        {"shape": [1, 32, 1024, 1024], "dim": -1, "dtype": "FP32", "descending": True},

        # ascending and descending variants
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "descending": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP32", "descending": True},
    ],
    "dtype_sweep": [
        # last-dim hidden sort
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "descending": False},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP16", "descending": False},
        {"shape": [128, 4096], "dim": -1, "dtype": "BF16", "descending": False},

        # feature-axis sort for 3D tensors
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP32", "descending": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP16", "descending": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "BF16", "descending": False},

        # attention score descending sort
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "FP32", "descending": True},
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "FP16", "descending": True},
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "BF16", "descending": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 32000],
        [8, 32000],
        [32, 32000],
        [64, 32000],
        [128, 32000],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [1024, 8192],
        [2048, 4096],
        [64, 256, 1024],
        [32, 512, 2048],
        [16, 1024, 4096],
        [8, 2048, 4096],
        [128, 128, 1024],
        [16, 256, 4096],
        [256, 64, 1024],
        [2, 8, 128, 128],
        [2, 16, 512, 512],
        [1, 32, 1024, 1024],
        [4, 32, 256, 256],
        [8, 16, 1024, 1024],
    ],
    "dim": [-1, 0, 1, 2, 3],
    "dtype": ["FP32", "FP16", "BF16"],
    "descending": [True, False],
}
