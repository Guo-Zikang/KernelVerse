"""Shape templates for mul benchmark cases."""

SINGLE_SHAPES = {
    "default": {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # same-shape elementwise multiply
        {"lhs_dims": [128, 2048], "rhs_dims": [128, 2048], "dtype": "FP32"},
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "FP32"},
        {"lhs_dims": [4096, 8192], "rhs_dims": [4096, 8192], "dtype": "FP32"},

        # row/column broadcast
        {"lhs_dims": [1024, 4096], "rhs_dims": [1, 4096], "dtype": "FP32"},
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 1], "dtype": "FP32"},
        {"lhs_dims": [8192, 4096], "rhs_dims": [1, 4096], "dtype": "FP32"},

        # hidden-dimension broadcast
        {"lhs_dims": [4, 512, 4096], "rhs_dims": [1, 1, 4096], "dtype": "FP32"},
        {"lhs_dims": [8, 1024, 4096], "rhs_dims": [1, 1024, 1], "dtype": "FP32"},
        {"lhs_dims": [16, 2048, 8192], "rhs_dims": [16, 1, 8192], "dtype": "FP32"},

        # 4D broadcast
        {"lhs_dims": [1, 32, 128, 128], "rhs_dims": [1, 1, 128, 128], "dtype": "FP32"},
        {"lhs_dims": [4, 32, 512, 128], "rhs_dims": [4, 32, 1, 128], "dtype": "FP32"},
        {"lhs_dims": [8, 16, 1024, 256], "rhs_dims": [1, 16, 1, 256], "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # same-shape elementwise multiply
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "FP32"},
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "FP16"},
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "BF16"},

        # broadcast multiply
        {"lhs_dims": [4096, 4096], "rhs_dims": [1, 4096], "dtype": "FP32"},
        {"lhs_dims": [4096, 4096], "rhs_dims": [1, 4096], "dtype": "FP16"},
        {"lhs_dims": [4096, 4096], "rhs_dims": [1, 4096], "dtype": "BF16"},

        # integer multiply
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "INT32"},
        {"lhs_dims": [1024, 4096], "rhs_dims": [1, 4096], "dtype": "INT32"},
        {"lhs_dims": [1024, 4096], "rhs_dims": [1024, 4096], "dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "lhs_dims": [
        [1, 2048],
        [16, 2048],
        [128, 2048],
        [512, 4096],
        [1024, 4096],
        [4096, 4096],
        [8192, 4096],
        [4, 512, 4096],
        [8, 1024, 4096],
        [16, 2048, 8192],
        [1, 32, 128, 128],
        [4, 32, 512, 128],
        [8, 16, 1024, 256],
    ],
    "rhs_dims": [
        [1],
        [1, 1],
        [1, 2048],
        [1, 4096],
        [1024, 1],
        [1, 1, 4096],
        [1, 1024, 1],
        [16, 1, 8192],
        [1, 1, 1, 1],
        [1, 1, 128, 128],
        [4, 32, 1, 128],
        [1, 16, 1, 256],
    ],
    "dtype": ["FP32", "FP16", "BF16", "INT32", "INT64"],
}
