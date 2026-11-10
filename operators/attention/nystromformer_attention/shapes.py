"""Shape templates for nystromformer_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "num_landmarks": 64,
        "pinv_iterations": 6,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 1, "seq_len": 256, "num_heads": 8, "head_dim": 64, "num_landmarks": 32, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_landmarks": 128, "pinv_iterations": 6, "dtype": "FP32"},

        # landmark count sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 32, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 128, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 256, "pinv_iterations": 6, "dtype": "FP32"},

        # pseudo-inverse iteration sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 2, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 4, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 8, "dtype": "FP32"},

        # batch and hidden size sweep
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 128, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},

        # dense fallback path when seq_len <= num_landmarks
        {"batch_size": 2, "seq_len": 128, "num_heads": 8, "head_dim": 64, "num_landmarks": 128, "pinv_iterations": 6, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "BF16"},

        # larger batch
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_landmarks": 64, "pinv_iterations": 6, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_landmarks": 128, "pinv_iterations": 6, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_landmarks": 128, "pinv_iterations": 6, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_landmarks": 128, "pinv_iterations": 6, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4],
    "seq_len": [128, 256, 512, 1024, 2048],
    "num_heads": [8, 16, 32],
    "head_dim": [64, 96, 128],
    "num_landmarks": [16, 32, 64, 128, 256],
    "pinv_iterations": [2, 4, 6, 8],
    "dtype": ["FP32", "FP16", "BF16"],
}
