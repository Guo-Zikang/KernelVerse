"""Shape templates for apply_token_bitmask benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": False},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # batch sweep
        {"batch": 1, "vocab_size": 32000, "dtype": "FP32", "use_indices": False},
        {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": False},
        {"batch": 128, "vocab_size": 32000, "dtype": "FP32", "use_indices": False},

        # vocab-size sweep
        {"batch": 16, "vocab_size": 8192, "dtype": "FP32", "use_indices": False},
        {"batch": 16, "vocab_size": 65536, "dtype": "FP32", "use_indices": False},
        {"batch": 8, "vocab_size": 128000, "dtype": "FP32", "use_indices": False},

        # selected-row sweep
        {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": True},
        {"batch": 128, "vocab_size": 32000, "dtype": "FP32", "use_indices": True},
        {"batch": 8, "vocab_size": 128000, "dtype": "FP32", "use_indices": True},
    ],
    "dtype_sweep": [
        # all rows
        {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": False},
        {"batch": 16, "vocab_size": 32000, "dtype": "FP16", "use_indices": False},
        {"batch": 16, "vocab_size": 32000, "dtype": "BF16", "use_indices": False},

        # selected rows
        {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": True},
        {"batch": 16, "vocab_size": 32000, "dtype": "FP16", "use_indices": True},
        {"batch": 16, "vocab_size": 32000, "dtype": "BF16", "use_indices": True},

        # larger vocab
        {"batch": 8, "vocab_size": 128000, "dtype": "FP32", "use_indices": False},
        {"batch": 8, "vocab_size": 128000, "dtype": "FP16", "use_indices": False},
        {"batch": 8, "vocab_size": 128000, "dtype": "BF16", "use_indices": False},
    ],
    "indices_sweep": [
        # row-selection behavior
        {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": False},
        {"batch": 16, "vocab_size": 32000, "dtype": "FP32", "use_indices": True},
        {"batch": 128, "vocab_size": 32000, "dtype": "FP32", "use_indices": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch": [1, 2, 4, 8, 16, 32, 64, 128],
    "vocab_size": [8192, 16000, 32000, 50016, 65536, 100000, 128000],
    "dtype": ["FP32", "FP16", "BF16"],
    "use_indices": [True, False],
}
