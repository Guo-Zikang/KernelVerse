"""Shape templates for scatter_reduce benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "input_shape": [128, 1024],
        "dim": 1,
        "reduce": "sum",
        "include_self": True,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # row and column scatter-reduce for 2D tensors
        {"input_shape": [512, 512], "dim": 0, "reduce": "sum", "include_self": True, "dtype": "FP32"},
        {"input_shape": [1024, 1024], "dim": 1, "reduce": "mean", "include_self": True, "dtype": "FP32"},
        {"input_shape": [2048, 4096], "dim": 0, "reduce": "amax", "include_self": False, "dtype": "FP32"},
        {"input_shape": [4096, 2048], "dim": -1, "reduce": "amin", "include_self": True, "dtype": "FP32"},

        # sequence and hidden dimension scatter-reduce for 3D tensors
        {"input_shape": [2, 128, 1024], "dim": 1, "reduce": "mean", "include_self": True, "dtype": "FP32"},
        {"input_shape": [4, 512, 4096], "dim": 2, "reduce": "sum", "include_self": False, "dtype": "FP32"},
        {"input_shape": [8, 1024, 4096], "dim": -1, "reduce": "amax", "include_self": True, "dtype": "FP32"},
        {"input_shape": [64, 1024, 128], "dim": 0, "reduce": "amin", "include_self": False, "dtype": "FP32"},

        # attention-like 4D layouts
        {"input_shape": [2, 8, 128, 64], "dim": 2, "reduce": "sum", "include_self": True, "dtype": "FP32"},
        {"input_shape": [4, 16, 512, 64], "dim": 1, "reduce": "mean", "include_self": False, "dtype": "FP32"},
        {"input_shape": [8, 32, 1024, 128], "dim": 3, "reduce": "amin", "include_self": True, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # 2D hidden scatter-reduce
        {"input_shape": [128, 4096], "dim": -1, "reduce": "sum", "include_self": True, "dtype": "FP32"},
        {"input_shape": [128, 4096], "dim": -1, "reduce": "sum", "include_self": True, "dtype": "FP16"},
        {"input_shape": [128, 4096], "dim": -1, "reduce": "sum", "include_self": True, "dtype": "BF16"},

        # 3D sequence scatter-reduce
        {"input_shape": [4, 512, 4096], "dim": 1, "reduce": "mean", "include_self": False, "dtype": "FP32"},
        {"input_shape": [4, 512, 4096], "dim": 1, "reduce": "mean", "include_self": False, "dtype": "FP16"},
        {"input_shape": [4, 512, 4096], "dim": 1, "reduce": "mean", "include_self": False, "dtype": "BF16"},

        # attention-like scatter-reduce
        {"input_shape": [2, 16, 512, 64], "dim": 2, "reduce": "amax", "include_self": True, "dtype": "FP32"},
        {"input_shape": [2, 16, 512, 64], "dim": 2, "reduce": "amax", "include_self": True, "dtype": "FP16"},
        {"input_shape": [2, 16, 512, 64], "dim": 2, "reduce": "amax", "include_self": True, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "input_shape": [
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
        [8, 128, 64],
        [16, 128, 64],
        [32, 512, 128],
        [64, 1024, 128],
        [128, 2048, 256],
        [16, 4096, 512],
        [2, 8, 128, 64],
        [4, 16, 512, 64],
        [8, 32, 1024, 128],
        [16, 16, 2048, 128],
        [2, 4, 1024, 128],
        [4, 8, 2048, 64],
    ],
    "dim": [-1, 0, 1, 2, 3],
    "reduce": ["sum", "mean", "amax", "amin"],
    "include_self": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
