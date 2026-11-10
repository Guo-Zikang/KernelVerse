"""Shape templates for lightning_attn_kv_reduce benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence and block-size sweep
        {"batch_size": 1, "num_heads": 8, "seq_len": 128, "head_dim": 64, "value_dim": 64, "block_size": 16, "dtype": "FP32"},
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "num_heads": 8, "seq_len": 2048, "head_dim": 64, "value_dim": 64, "block_size": 128, "dtype": "FP32"},

        # head/value dimension sweep
        {"batch_size": 1, "num_heads": 16, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "num_heads": 16, "seq_len": 512, "head_dim": 128, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "num_heads": 16, "seq_len": 512, "head_dim": 64, "value_dim": 128, "block_size": 64, "dtype": "FP32"},

        # batch and head-count sweep
        {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 4, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "num_heads": 32, "seq_len": 256, "head_dim": 64, "value_dim": 64, "block_size": 32, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 64, "block_size": 64, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "num_heads": 16, "seq_len": 2048, "head_dim": 64, "value_dim": 64, "block_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "num_heads": 16, "seq_len": 2048, "head_dim": 64, "value_dim": 64, "block_size": 128, "dtype": "FP16"},
        {"batch_size": 1, "num_heads": 16, "seq_len": 2048, "head_dim": 64, "value_dim": 64, "block_size": 128, "dtype": "BF16"},

        # larger value dimension
        {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 128, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "value_dim": 128, "block_size": 64, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "num_heads": [4, 8, 16, 32],
    "seq_len": [128, 256, 512, 1024, 2048],
    "head_dim": [32, 64, 128],
    "value_dim": [32, 64, 128],
    "block_size": [16, 32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
