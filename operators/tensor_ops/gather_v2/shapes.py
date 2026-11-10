"""Shape templates for embedding gather benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "params_shape": [151936, 2048],
        "indices_shape": [1, 128],
        "axis": 0,
        "dtype": "BF16",
        "indices_dtype": "INT64",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # production-style embedding lookup
        {"params_shape": [151936, 2048], "indices_shape": [1, 128], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [151936, 2048], "indices_shape": [1, 128], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [32000, 4096], "indices_shape": [8, 128], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},

        # smaller embedding tables
        {"params_shape": [4096, 1024], "indices_shape": [32], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [8192, 2048], "indices_shape": [16, 64], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [16384, 1024], "indices_shape": [4, 256], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},

        # nonzero gather axis
        {"params_shape": [128, 4096, 64], "indices_shape": [32], "axis": 1, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [16, 512, 4096], "indices_shape": [8, 64], "axis": -1, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [8, 32, 1024, 128], "indices_shape": [16], "axis": 2, "dtype": "FP32", "indices_dtype": "INT64"},
    ],
    "dtype_sweep": [
        # embedding lookup
        {"params_shape": [32000, 4096], "indices_shape": [8, 128], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [32000, 4096], "indices_shape": [8, 128], "axis": 0, "dtype": "FP16", "indices_dtype": "INT64"},
        {"params_shape": [32000, 4096], "indices_shape": [8, 128], "axis": 0, "dtype": "BF16", "indices_dtype": "INT64"},

        # small batched lookup
        {"params_shape": [8192, 2048], "indices_shape": [16, 64], "axis": 0, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [8192, 2048], "indices_shape": [16, 64], "axis": 0, "dtype": "FP16", "indices_dtype": "INT64"},
        {"params_shape": [8192, 2048], "indices_shape": [16, 64], "axis": 0, "dtype": "BF16", "indices_dtype": "INT64"},

        # nonzero gather axis
        {"params_shape": [128, 4096, 64], "indices_shape": [32], "axis": 1, "dtype": "FP32", "indices_dtype": "INT64"},
        {"params_shape": [128, 4096, 64], "indices_shape": [32], "axis": 1, "dtype": "FP16", "indices_dtype": "INT64"},
        {"params_shape": [128, 4096, 64], "indices_shape": [32], "axis": 1, "dtype": "BF16", "indices_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "params_shape": [
        [4096, 1024],
        [8192, 2048],
        [16384, 1024],
        [32000, 4096],
        [151936, 2048],
        [128, 4096, 64],
        [16, 512, 4096],
        [8, 32, 1024, 128],
    ],
    "indices_shape": [[1], [32], [1, 128], [8, 128], [16, 64], [4, 256]],
    "axis": [-1, 0, 1, 2],
    "dtype": ["FP32", "FP16", "BF16"],
    "indices_dtype": ["INT32", "INT64"],
}
