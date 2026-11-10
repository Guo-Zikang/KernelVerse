"""Shape templates for top_p_renorm_probs benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 128, "vocab_size": 4096, "top_p": 0.9, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # batch size sweep
        {"batch_size": 1, "vocab_size": 4096, "top_p": 0.9, "dtype": "FP32"},
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.9, "dtype": "FP32"},
        {"batch_size": 512, "vocab_size": 4096, "top_p": 0.9, "dtype": "FP32"},

        # vocabulary size sweep
        {"batch_size": 128, "vocab_size": 1024, "top_p": 0.9, "dtype": "FP32"},
        {"batch_size": 128, "vocab_size": 8192, "top_p": 0.9, "dtype": "FP32"},
        {"batch_size": 64, "vocab_size": 32000, "top_p": 0.9, "dtype": "FP32"},

        # top-p sweep
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.1, "dtype": "FP32"},
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.5, "dtype": "FP32"},
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.95, "dtype": "FP32"},
        {"batch_size": 128, "vocab_size": 4096, "top_p": 1.0, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard top-p renormalization
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.9, "dtype": "FP32"},
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.9, "dtype": "FP16"},
        {"batch_size": 128, "vocab_size": 4096, "top_p": 0.9, "dtype": "BF16"},

        # large vocabulary
        {"batch_size": 64, "vocab_size": 32000, "top_p": 0.95, "dtype": "FP32"},
        {"batch_size": 64, "vocab_size": 32000, "top_p": 0.95, "dtype": "FP16"},
        {"batch_size": 64, "vocab_size": 32000, "top_p": 0.95, "dtype": "BF16"},

        # decode-style batch
        {"batch_size": 16, "vocab_size": 4096, "top_p": 0.8, "dtype": "FP32"},
        {"batch_size": 16, "vocab_size": 4096, "top_p": 0.8, "dtype": "FP16"},
        {"batch_size": 16, "vocab_size": 4096, "top_p": 0.8, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 8, 16, 32, 64, 128, 256, 512],
    "vocab_size": [128, 256, 1024, 2048, 4096, 8192, 32000],
    "top_p": [0.1, 0.5, 0.8, 0.9, 0.95, 1.0],
    "dtype": ["FP32", "FP16", "BF16"],
}
