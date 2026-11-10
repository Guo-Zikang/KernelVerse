"""Shape templates for L2 normalization benchmark cases."""

SINGLE_SHAPES = {
    "default": {"shape": [128, 1024], "eps": 1e-6, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # embedding vector sweep
        {"shape": [32, 256], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [128, 768], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [128, 1024], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [128, 4096], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [512, 4096], "eps": 1e-6, "dtype": "FP32"},

        # batched sequence layouts
        {"shape": [2, 128, 1024], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [8, 1024, 4096], "eps": 1e-6, "dtype": "FP32"},

        # multi-dimensional layouts
        {"shape": [2, 16, 64, 128], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [4, 32, 32, 256], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [8, 8, 128, 512], "eps": 1e-6, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # embedding vector
        {"shape": [128, 1024], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [128, 1024], "eps": 1e-6, "dtype": "FP16"},
        {"shape": [128, 1024], "eps": 1e-6, "dtype": "BF16"},

        # LLM hidden vector
        {"shape": [128, 4096], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [128, 4096], "eps": 1e-6, "dtype": "FP16"},
        {"shape": [128, 4096], "eps": 1e-6, "dtype": "BF16"},

        # batched sequence layout
        {"shape": [4, 512, 4096], "eps": 1e-6, "dtype": "FP32"},
        {"shape": [4, 512, 4096], "eps": 1e-6, "dtype": "FP16"},
        {"shape": [4, 512, 4096], "eps": 1e-6, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 128],
        [1, 512],
        [1, 1024],
        [1, 4096],
        [32, 256],
        [128, 768],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [2048, 1024],
        [2, 128, 1024],
        [4, 512, 4096],
        [8, 1024, 4096],
        [16, 1, 4096],
        [128, 1, 8192],
        [2, 16, 64, 128],
        [4, 32, 32, 256],
        [8, 8, 128, 512],
        [16, 16, 64, 1024],
        [2, 4, 1024, 2048],
        [4096, 768],
        [1024, 8192],
        [2, 2048, 4096],
        [16, 256, 2048],
        [4, 16, 1024, 512],
    ],
    "eps": [1e-6, 1e-5],
    "dtype": ["FP32", "FP16", "BF16"],
}
