"""Shape templates for linear projection benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "input_shape": [128, 1024],
        "out_features": 4096,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # square
        {"input_shape": [1024, 1024], "out_features": 1024, "dtype": "FP32"},
        {"input_shape": [2048, 2048], "out_features": 2048, "dtype": "FP32"},
        {"input_shape": [4096, 4096], "out_features": 4096, "dtype": "FP32"},

        # LLM projection style
        {"input_shape": [128, 4096], "out_features": 4096, "dtype": "FP32"},
        {"input_shape": [512, 4096], "out_features": 11008, "dtype": "FP32"},
        {"input_shape": [2048, 4096], "out_features": 11008, "dtype": "FP32"},

        # extreme aspect ratio
        {"input_shape": [1, 4096], "out_features": 4096, "dtype": "FP32"},
        {"input_shape": [8, 4096], "out_features": 11008, "dtype": "FP32"},
        {"input_shape": [4096, 8192], "out_features": 1, "dtype": "FP32"},

        # unaligned M or N
        {"input_shape": [1008, 1024], "out_features": 1024, "dtype": "FP32"},
        {"input_shape": [1024, 1024], "out_features": 2060, "dtype": "FP32"},
        {"input_shape": [2048, 4096], "out_features": 5120, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # square large
        {"input_shape": [1024, 1024], "out_features": 1024, "dtype": "FP32"},
        {"input_shape": [1024, 1024], "out_features": 1024, "dtype": "FP16"},
        {"input_shape": [1024, 1024], "out_features": 1024, "dtype": "BF16"},

        # projection
        {"input_shape": [512, 4096], "out_features": 11008, "dtype": "FP32"},
        {"input_shape": [512, 4096], "out_features": 11008, "dtype": "FP16"},
        {"input_shape": [512, 4096], "out_features": 11008, "dtype": "BF16"},

        # decode small M
        {"input_shape": [8, 4096], "out_features": 11008, "dtype": "FP32"},
        {"input_shape": [8, 4096], "out_features": 11008, "dtype": "FP16"},
        {"input_shape": [8, 4096], "out_features": 11008, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "input_shape": [
        [1, 128],
        [1, 512],
        [2, 256, 512],
        [8, 128, 1024],
        [16, 512, 4096],
        [16, 768],
        [4, 2560],
        [8, 3072],
        [8, 4096],
        [1, 4096],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [1024, 1024],
        [2048, 2048],
        [2, 128, 1024],
        [4, 512, 4096],
        [8, 1024, 4096],
        [2, 2048, 4096],
        [16, 8192],
        [4096, 1],
        [1024, 8],
    ],
    "out_features": [1, 8, 128, 256, 512, 768, 1024, 2560, 3072, 4096, 6144],
    "dtype": ["FP32", "FP16", "BF16"],
}
