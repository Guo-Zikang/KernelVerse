"""Shape templates for block_sparse_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 2,
        "seq_len": 512,
        "num_heads": 8,
        "head_dim": 64,
        "block_size": 32,
        "num_selected_blocks": 4,
        "selection_mode": "fixed",
        "random_seed": 0,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # context length sweep
        {"batch_size": 1, "seq_len": 256, "num_heads": 8, "head_dim": 64, "block_size": 16, "num_selected_blocks": 2, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 6, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},

        # selection mode sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "random", "random_seed": 17, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "local_plus_topk", "random_seed": 0, "dtype": "FP32"},

        # sparsity and block size sweep
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 16, "num_selected_blocks": 2, "selection_mode": "local_plus_topk", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "local_plus_topk", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 8, "selection_mode": "local_plus_topk", "random_seed": 0, "dtype": "FP32"},

        # hidden size and batch sweep
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 40, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "fixed", "random_seed": 0, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 6, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 6, "selection_mode": "fixed", "random_seed": 0, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_selected_blocks": 6, "selection_mode": "fixed", "random_seed": 0, "dtype": "BF16"},

        # large hidden
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "block_size": 32, "num_selected_blocks": 4, "selection_mode": "topk", "random_seed": 0, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4],
    "seq_len": [128, 256, 512, 1024, 2048],
    "num_heads": [8, 16, 32],
    "head_dim": [64, 96, 128],
    "block_size": [16, 32, 64],
    "num_selected_blocks": [1, 2, 4, 8],
    "selection_mode": ["topk", "fixed", "random", "local_plus_topk"],
    "random_seed": [0, 17, 23],
    "dtype": ["FP32", "FP16", "BF16"],
}
