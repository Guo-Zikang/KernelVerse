"""Shape templates for topk benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "k": 16,
        "dim": -1,
        "dtype": "FP32",
        "largest": True,
        "sorted": True,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # last-dim top-k over vocab or hidden dimensions
        {"shape": [1, 32000], "k": 50, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [8, 32000], "k": 100, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [32, 32000], "k": 100, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [128, 4096], "k": 16, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},

        # feature-axis top-k for 3D tensors
        {"shape": [64, 256, 1024], "k": 16, "dim": 1, "dtype": "FP32", "largest": False, "sorted": True},
        {"shape": [32, 512, 2048], "k": 32, "dim": 1, "dtype": "FP32", "largest": False, "sorted": True},
        {"shape": [16, 1024, 4096], "k": 64, "dim": 1, "dtype": "FP32", "largest": False, "sorted": True},

        # attention score layouts
        {"shape": [2, 8, 128, 128], "k": 8, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [2, 16, 512, 512], "k": 32, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [1, 32, 1024, 1024], "k": 64, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},

        # unsorted top-k variants
        {"shape": [128, 4096], "k": 32, "dim": -1, "dtype": "FP32", "largest": True, "sorted": False},
        {"shape": [32, 512, 2048], "k": 32, "dim": 1, "dtype": "FP32", "largest": False, "sorted": False},
    ],
    "dtype_sweep": [
        # last-dim hidden top-k
        {"shape": [128, 4096], "k": 16, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [128, 4096], "k": 16, "dim": -1, "dtype": "FP16", "largest": True, "sorted": True},
        {"shape": [128, 4096], "k": 16, "dim": -1, "dtype": "BF16", "largest": True, "sorted": True},

        # feature-axis top-k for 3D tensors
        {"shape": [32, 512, 2048], "k": 32, "dim": 1, "dtype": "FP32", "largest": False, "sorted": True},
        {"shape": [32, 512, 2048], "k": 32, "dim": 1, "dtype": "FP16", "largest": False, "sorted": True},
        {"shape": [32, 512, 2048], "k": 32, "dim": 1, "dtype": "BF16", "largest": False, "sorted": True},

        # attention score layout
        {"shape": [2, 16, 512, 512], "k": 32, "dim": -1, "dtype": "FP32", "largest": True, "sorted": True},
        {"shape": [2, 16, 512, 512], "k": 32, "dim": -1, "dtype": "FP16", "largest": True, "sorted": True},
        {"shape": [2, 16, 512, 512], "k": 32, "dim": -1, "dtype": "BF16", "largest": True, "sorted": True},
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
    "k": [1, 2, 4, 8, 16, 32, 64, 100],
    "dim": [-1, 0, 1, 2, 3],
    "dtype": ["FP32", "FP16", "BF16"],
    "largest": [True, False],
    "sorted": [True, False],
}
