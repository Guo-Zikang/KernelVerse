"""Shape templates for multinomial sampling benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 1024],
        "num_samples": 1,
        "replacement": False,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # single distribution over vocabulary
        {"shape": [32000], "num_samples": 1, "replacement": False, "dtype": "FP32"},
        {"shape": [32000], "num_samples": 16, "replacement": False, "dtype": "FP32"},
        {"shape": [32000], "num_samples": 32, "replacement": True, "dtype": "FP32"},

        # batched distributions over vocabulary
        {"shape": [1, 32000], "num_samples": 1, "replacement": False, "dtype": "FP32"},
        {"shape": [8, 32000], "num_samples": 4, "replacement": False, "dtype": "FP32"},
        {"shape": [32, 32000], "num_samples": 8, "replacement": True, "dtype": "FP32"},
        {"shape": [128, 32000], "num_samples": 1, "replacement": False, "dtype": "FP32"},

        # small and medium vocabularies
        {"shape": [128, 128], "num_samples": 8, "replacement": False, "dtype": "FP32"},
        {"shape": [512, 256], "num_samples": 16, "replacement": False, "dtype": "FP32"},
        {"shape": [1024, 512], "num_samples": 32, "replacement": True, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # default batched distribution
        {"shape": [128, 1024], "num_samples": 1, "replacement": False, "dtype": "FP32"},
        {"shape": [128, 1024], "num_samples": 1, "replacement": False, "dtype": "FP16"},
        {"shape": [128, 1024], "num_samples": 1, "replacement": False, "dtype": "BF16"},

        # multiple samples without replacement
        {"shape": [8, 32000], "num_samples": 4, "replacement": False, "dtype": "FP32"},
        {"shape": [8, 32000], "num_samples": 4, "replacement": False, "dtype": "FP16"},
        {"shape": [8, 32000], "num_samples": 4, "replacement": False, "dtype": "BF16"},

        # multiple samples with replacement
        {"shape": [32, 32000], "num_samples": 8, "replacement": True, "dtype": "FP32"},
        {"shape": [32, 32000], "num_samples": 8, "replacement": True, "dtype": "FP16"},
        {"shape": [32, 32000], "num_samples": 8, "replacement": True, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [32000],
        [4096],
        [128, 128],
        [512, 256],
        [1024, 512],
        [128, 1024],
        [64, 2048],
        [1, 32000],
        [8, 32000],
        [16, 32000],
        [32, 32000],
        [128, 32000],
        [256, 4096],
        [512, 4096],
        [512, 8192],
        [1024, 16000],
        [2048, 8192],
    ],
    "num_samples": [1, 2, 4, 8, 16, 32],
    "replacement": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
