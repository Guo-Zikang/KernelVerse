"""Shape templates for dynamic_scaled_int8_quant benchmark cases."""

SINGLE_SHAPES = {
    "default": {"dims": [128, 4096], "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # hidden size sweep
        {"dims": [1, 1024], "dtype": "FP32"},
        {"dims": [128, 1024], "dtype": "FP32"},
        {"dims": [128, 4096], "dtype": "FP32"},
        {"dims": [512, 4096], "dtype": "FP32"},
        {"dims": [512, 8192], "dtype": "FP32"},

        # batched sequence flattened as token rows
        {"dims": [256, 1024], "dtype": "FP32"},
        {"dims": [2048, 4096], "dtype": "FP32"},
        {"dims": [8192, 4096], "dtype": "FP32"},

        # higher-rank tensor layouts
        {"dims": [2, 128, 1024], "dtype": "FP32"},
        {"dims": [4, 512, 4096], "dtype": "FP32"},
        {"dims": [8, 1024, 4096], "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # hidden vector
        {"dims": [128, 4096], "dtype": "FP32"},
        {"dims": [128, 4096], "dtype": "FP16"},
        {"dims": [128, 4096], "dtype": "BF16"},

        # batched sequence
        {"dims": [512, 4096], "dtype": "FP32"},
        {"dims": [512, 4096], "dtype": "FP16"},
        {"dims": [512, 4096], "dtype": "BF16"},

        # decode-style tensors
        {"dims": [16, 4096], "dtype": "FP32"},
        {"dims": [16, 4096], "dtype": "FP16"},
        {"dims": [16, 4096], "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "dims": [
        [1, 1024],
        [1, 4096],
        [1, 8192],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [2048, 4096],
        [512, 8192],
        [16, 4096],
        [128, 8192],
        [256, 4096],
        [1024, 1024],
        [1024, 4096],
        [4096, 1024],
        [8192, 512],
        [32768, 256],
        [65536, 128],
        [2, 128, 1024],
        [4, 512, 4096],
        [8, 1024, 4096],
        [16384, 1024],
        [32768, 512],
        [2, 2048, 4096],
        [4, 1024, 8192],
        [2, 16, 1024, 256],
    ],
    "dtype": ["FP32", "FP16", "BF16"],
}
