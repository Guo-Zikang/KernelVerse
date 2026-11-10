"""Shape templates for block local attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 2,
        "seq_len": 512,
        "num_heads": 8,
        "head_dim": 64,
        "block_size": 64,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # context length sweep
        {"batch_size": 1, "seq_len": 256, "num_heads": 8, "head_dim": 64, "block_size": 32, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 8, "head_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 128, "dtype": "FP32"},

        # block size sweep
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 16, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 128, "dtype": "FP32"},

        # hidden size and batch sweep
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 40, "head_dim": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 64, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 64, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 128, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 128, "dtype": "BF16"},

        # large hidden
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 64, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "seq_len": [128, 256, 512, 1024, 2048],
    "num_heads": [8, 16, 32, 40, 64],
    "head_dim": [64, 96, 128],
    "block_size": [16, 32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
