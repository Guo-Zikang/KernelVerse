"""Shape templates for matmul_v2 benchmark cases.

matmul_v2 uses the projection-weight layout:
- tensor_a: [M, K]
- tensor_b: [N, K]
- output: [M, N], computed as tensor_a @ tensor_b.T
"""

SINGLE_SHAPES = {
    "default": {
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # square
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32"},
        {"M": 2048, "N": 2048, "K": 2048, "dtype": "FP32"},
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "FP32"},

        # LLM projection style
        {"M": 128, "N": 4096, "K": 4096, "dtype": "FP32"},
        {"M": 512, "N": 11008, "K": 4096, "dtype": "FP32"},
        {"M": 2048, "N": 11008, "K": 4096, "dtype": "FP32"},

        # extreme aspect ratio
        {"M": 1, "N": 4096, "K": 4096, "dtype": "FP32"},
        {"M": 8, "N": 11008, "K": 4096, "dtype": "FP32"},
        {"M": 4096, "N": 1, "K": 8192, "dtype": "FP32"},

        # unaligned M or N
        {"M": 1008, "N": 1024, "K": 1024, "dtype": "FP32"},
        {"M": 1024, "N": 2060, "K": 1024, "dtype": "FP32"},
        {"M": 2048, "N": 5120, "K": 4096, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # square large
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP32"},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "FP16"},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": "BF16"},

        # projection
        {"M": 512, "N": 11008, "K": 4096, "dtype": "FP32"},
        {"M": 512, "N": 11008, "K": 4096, "dtype": "FP16"},
        {"M": 512, "N": 11008, "K": 4096, "dtype": "BF16"},

        # decode small M
        {"M": 8, "N": 11008, "K": 4096, "dtype": "FP32"},
        {"M": 8, "N": 11008, "K": 4096, "dtype": "FP16"},
        {"M": 8, "N": 11008, "K": 4096, "dtype": "BF16"},
    ]
}

RANDOM_SHAPE_SPACE = {
    "M": [1, 8, 512, 1024, 2048, 3072, 4096, 8192, 10240, 20736, 65536],
    "N": [1, 8, 512, 1024, 2560, 3072, 4096, 6144, 8192, 9728, 11008],
    "K": [1, 8, 256, 512, 1024, 1536, 2048, 2816, 3584, 4096, 5120],
    "dtype": ["FP32", "FP16", "BF16"],
}
