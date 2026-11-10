"""Shape templates for baddbmm benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 16,
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "dtype": "FP32",
        "alpha": 1.0,
        "beta": 1.0,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # square
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 2048, "N": 2048, "K": 2048, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 4096, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # LLM projection style
        {"batch_size": 16, "M": 128, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 512, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 2048, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # extreme aspect ratio
        {"batch_size": 64, "M": 1, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 64, "M": 8, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 64, "M": 4096, "N": 1, "K": 8192, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # unaligned M or N
        {"batch_size": 16, "M": 1008, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 1024, "N": 2060, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 2048, "N": 5120, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # batch size sweep
        {"batch_size": 1, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 4, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 32, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 64, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # alpha/beta sweep
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 0.5},
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 0.5, "beta": 0.5},
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 0.0, "beta": 1.0},
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 0.0},

        # alpha/beta sweep on larger shape
        {"batch_size": 16, "M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 0.5},
        {"batch_size": 16, "M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 0.5, "beta": 0.5},
        {"batch_size": 16, "M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 0.0, "beta": 1.0},
        {"batch_size": 16, "M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 0.0},
    ],
    "dtype_sweep": [
        # square large
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "FP16", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 1024, "N": 1024, "K": 1024, "dtype": "BF16", "alpha": 1.0, "beta": 1.0},

        # projection
        {"batch_size": 16, "M": 512, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 512, "N": 11008, "K": 4096, "dtype": "FP16", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 16, "M": 512, "N": 11008, "K": 4096, "dtype": "BF16", "alpha": 1.0, "beta": 1.0},

        # decode small M
        {"batch_size": 32, "M": 8, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 32, "M": 8, "N": 11008, "K": 4096, "dtype": "FP16", "alpha": 1.0, "beta": 1.0},
        {"batch_size": 32, "M": 8, "N": 11008, "K": 4096, "dtype": "BF16", "alpha": 1.0, "beta": 1.0},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16, 32, 64],
    "M": [1, 8, 128, 256, 512, 768, 1024, 2048, 3072, 4096, 8192],
    "N": [1, 8, 128, 256, 512, 768, 1024, 2560, 3072, 4096, 6144],
    "K": [1, 8, 128, 256, 512, 1024, 1536, 2048, 2816, 3584, 4096],
    "dtype": ["FP32", "FP16", "BF16"],
    "alpha": [-1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0],
    "beta": [0.0, 0.5, 1.0, 2.0],
}
