"""Shape templates for chunk_kda_scaled_dot_kkt_fwd benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 4, "seq_len": 128, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 2, "seq_len": 32, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 128, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 256, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},

        # chunk size sweep
        {"batch_size": 4, "seq_len": 128, "num_heads": 8, "key_dim": 64, "chunk_size": 32, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 256, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "key_dim": 64, "chunk_size": 128, "dtype": "FP32"},

        # head and key dimension sweep
        {"batch_size": 4, "seq_len": 128, "num_heads": 4, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 128, "num_heads": 16, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 256, "num_heads": 8, "key_dim": 128, "chunk_size": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard KKT stage
        {"batch_size": 4, "seq_len": 128, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 128, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 128, "num_heads": 8, "key_dim": 64, "chunk_size": 64, "dtype": "BF16"},

        # wider key dimension
        {"batch_size": 2, "seq_len": 256, "num_heads": 8, "key_dim": 128, "chunk_size": 128, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 256, "num_heads": 8, "key_dim": 128, "chunk_size": 128, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 256, "num_heads": 8, "key_dim": 128, "chunk_size": 128, "dtype": "BF16"},

        # decode-like shorter sequence
        {"batch_size": 16, "seq_len": 32, "num_heads": 8, "key_dim": 64, "chunk_size": 32, "dtype": "FP32"},
        {"batch_size": 16, "seq_len": 32, "num_heads": 8, "key_dim": 64, "chunk_size": 32, "dtype": "FP16"},
        {"batch_size": 16, "seq_len": 32, "num_heads": 8, "key_dim": 64, "chunk_size": 32, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "seq_len": [16, 32, 64, 128, 256, 512],
    "num_heads": [1, 4, 8, 16],
    "key_dim": [32, 64, 128],
    "chunk_size": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
