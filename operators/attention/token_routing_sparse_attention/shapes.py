"""Shape templates for token_routing_sparse_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "num_routes": 32,
        "top_k_routes": 2,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 2, "seq_len": 256, "num_heads": 16, "head_dim": 64, "num_routes": 16, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_routes": 64, "top_k_routes": 4, "dtype": "FP32"},

        # route count sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 8, "top_k_routes": 1, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 16, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 64, "top_k_routes": 4, "dtype": "FP32"},

        # top-k route sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 1, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 4, "dtype": "FP32"},

        # batch and hidden size sweep
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 64, "num_routes": 64, "top_k_routes": 4, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 128, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "BF16"},

        # larger batch
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_routes": 32, "top_k_routes": 2, "dtype": "BF16"},

        # sparse route setting
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_routes": 64, "top_k_routes": 4, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_routes": 64, "top_k_routes": 4, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_routes": 64, "top_k_routes": 4, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4],
    "seq_len": [256, 512, 1024, 2048],
    "num_heads": [8, 16, 32],
    "head_dim": [64, 96, 128],
    "num_routes": [8, 16, 32, 64],
    "top_k_routes": [1, 2, 4],
    "dtype": ["FP32", "FP16", "BF16"],
}
