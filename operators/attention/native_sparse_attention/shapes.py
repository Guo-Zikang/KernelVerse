"""Shape templates for native_sparse_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 512,
        "num_heads": 16,
        "head_dim": 64,
        "block_size": 32,
        "num_selected_blocks": 4,
        "local_window_size": 128,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 1, "seq_len": 256, "num_heads": 8, "head_dim": 64, "block_size": 16, "num_selected_blocks": 2, "local_window_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},

        # block size sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 16, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},

        # selected block count sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 1, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 2, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},

        # local window sweep
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 32, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},

        # batch and hidden size sweep
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "BF16"},

        # larger batch
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "BF16"},

        # block-size variant
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 4, "local_window_size": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2],
    "seq_len": [256, 512, 1024],
    "num_heads": [8, 16],
    "head_dim": [64],
    "block_size": [16, 32],
    "num_selected_blocks": [1, 2, 4],
    "local_window_size": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
