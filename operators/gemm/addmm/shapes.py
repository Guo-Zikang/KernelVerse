"""Shape templates for addmm benchmark cases."""

SINGLE_SHAPES = {
    "default": {
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
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 2048, "N": 2048, "K": 2048, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # LLM projection style
        {"M": 128, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 512, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 2048, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # extreme aspect ratio
        {"M": 1, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 8, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 4096, "N": 1, "K": 8192, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # unaligned M or N
        {"M": 1008, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 1024, "N": 2060, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 2048, "N": 5120, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},

        # alpha/beta sweep
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 0.5},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 0.5, "beta": 0.5},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 0.0, "beta": 1.0},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 0.0},

        # alpha/beta sweep on larger shape
        {"M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 0.5},
        {"M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 0.5, "beta": 0.5},
        {"M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 0.0, "beta": 1.0},
        {"M": 2048, "N": 4096, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 0.0},
    ],
    "dtype_sweep": [
        # square large
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP16", "alpha": 1.0, "beta": 1.0},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "BF16", "alpha": 1.0, "beta": 1.0},

        # projection
        {"M": 512, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 512, "N": 11008, "K": 4096, "dtype": "FP16", "alpha": 1.0, "beta": 1.0},
        {"M": 512, "N": 11008, "K": 4096, "dtype": "BF16", "alpha": 1.0, "beta": 1.0},

        # decode small M
        {"M": 8, "N": 11008, "K": 4096, "dtype": "FP32", "alpha": 1.0, "beta": 1.0},
        {"M": 8, "N": 11008, "K": 4096, "dtype": "FP16", "alpha": 1.0, "beta": 1.0},
        {"M": 8, "N": 11008, "K": 4096, "dtype": "BF16", "alpha": 1.0, "beta": 1.0},
    ]
}

RANDOM_SHAPE_SPACE = {
    "M": [1, 8, 512, 1024, 2048, 3072, 4096, 8192, 10240, 20736, 65536],
    "N": [1, 8, 512, 1024, 2560, 3072, 4096, 6144, 8192, 9728, 11008],
    "K": [1, 8, 256, 512, 1024, 1536, 2048, 2816, 3584, 4096, 5120],
    "dtype": ["FP32", "FP16", "BF16"],
    "alpha": [-1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0],
    "beta": [0.0, 0.5, 1.0, 1.0, 2.0],
}
