"""Shape templates for max benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "dim": -1,
        "dtype": "FP32",
        "keepdim": False,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # last-dim reduction over vocab or hidden dimensions
        {"shape": [1, 32000], "dim": -1, "dtype": "FP32", "keepdim": False},
        {"shape": [8, 32000], "dim": -1, "dtype": "FP32", "keepdim": False},
        {"shape": [32, 32000], "dim": -1, "dtype": "FP32", "keepdim": True},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "keepdim": False},

        # feature-axis reduction for 3D tensors
        {"shape": [64, 256, 1024], "dim": 1, "dtype": "FP32", "keepdim": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP32", "keepdim": False},
        {"shape": [16, 1024, 4096], "dim": 1, "dtype": "FP32", "keepdim": True},

        # attention score layouts
        {"shape": [2, 8, 128, 128], "dim": -1, "dtype": "FP32", "keepdim": False},
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "FP32", "keepdim": False},
        {"shape": [1, 32, 1024, 1024], "dim": -1, "dtype": "FP32", "keepdim": True},
    ],
    "dtype_sweep": [
        # last-dim hidden reduction
        {"shape": [128, 4096], "dim": -1, "dtype": "FP32", "keepdim": False},
        {"shape": [128, 4096], "dim": -1, "dtype": "FP16", "keepdim": False},
        {"shape": [128, 4096], "dim": -1, "dtype": "BF16", "keepdim": False},

        # feature-axis reduction for 3D tensors
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP32", "keepdim": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "FP16", "keepdim": False},
        {"shape": [32, 512, 2048], "dim": 1, "dtype": "BF16", "keepdim": False},

        # attention score layout with keepdim enabled
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "FP32", "keepdim": True},
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "FP16", "keepdim": True},
        {"shape": [2, 16, 512, 512], "dim": -1, "dtype": "BF16", "keepdim": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 32000],
        [8, 32000],
        [32, 32000],
        [64, 32000],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [1024, 8192],
        [64, 256, 1024],
        [32, 512, 2048],
        [16, 1024, 4096],
        [8, 2048, 4096],
        [128, 128, 1024],
        [2, 8, 128, 128],
        [2, 16, 512, 512],
        [1, 32, 1024, 1024],
        [4, 32, 256, 256],
    ],
    "dim": [-1, 0, 1, 2, 3],
    "dtype": ["FP32", "FP16", "BF16"],
    "keepdim": [True, False],
}
