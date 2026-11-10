"""Shape templates for int8_scaled_mm_azp benchmark cases."""

SINGLE_SHAPES = {
    "default": {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # M sweep
        {"M": 128, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 8192, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},

        # projection shape sweep
        {"M": 1024, "N": 2048, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 1024, "N": 11008, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},

        # K sweep
        {"M": 1024, "N": 4096, "K": 2048, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 1024, "N": 4096, "K": 8192, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 2048, "N": 8192, "K": 8192, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
    ],
    "dtype_sweep": [
        # AZP and bias
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP32"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "FP16"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": True, "output_dtype": "BF16"},

        # AZP without bias
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": False, "output_dtype": "FP32"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": False, "output_dtype": "FP16"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": True, "has_bias": False, "output_dtype": "BF16"},

        # bias without AZP
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": False, "has_bias": True, "output_dtype": "FP32"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": False, "has_bias": True, "output_dtype": "FP16"},
        {"M": 1024, "N": 4096, "K": 4096, "has_azp": False, "has_bias": True, "output_dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "M": [1, 8, 32, 128, 512, 1024, 2048, 8192],
    "N": [512, 1024, 2048, 4096, 6144, 8192, 11008],
    "K": [512, 1024, 2048, 4096, 6144, 8192],
    "has_azp": [True, False],
    "has_bias": [True, False],
    "output_dtype": ["FP32", "FP16", "BF16"],
}
