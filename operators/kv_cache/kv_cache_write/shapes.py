"""Shape templates for KV cache write benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 4,
        "num_kv_heads": 8,
        "max_seq_len": 2048,
        "write_seq_len": 1,
        "head_dim": 128,
        "position_mode": "shared",
        "dtype": "FP16",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode write sweep
        {"batch_size": 1, "num_kv_heads": 8, "max_seq_len": 2048, "write_seq_len": 1, "head_dim": 128, "position_mode": "shared", "dtype": "FP32"},
        {"batch_size": 8, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 1, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP32"},
        {"batch_size": 16, "num_kv_heads": 16, "max_seq_len": 8192, "write_seq_len": 1, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP32"},

        # prefill chunk write sweep
        {"batch_size": 2, "num_kv_heads": 8, "max_seq_len": 2048, "write_seq_len": 128, "head_dim": 128, "position_mode": "shared", "dtype": "FP32"},
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 256, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP32"},
        {"batch_size": 8, "num_kv_heads": 16, "max_seq_len": 8192, "write_seq_len": 512, "head_dim": 128, "position_mode": "shared", "dtype": "FP32"},

        # head dimension sweep
        {"batch_size": 4, "num_kv_heads": 4, "max_seq_len": 2048, "write_seq_len": 32, "head_dim": 64, "position_mode": "shared", "dtype": "FP32"},
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 32, "head_dim": 128, "position_mode": "shared", "dtype": "FP32"},
        {"batch_size": 2, "num_kv_heads": 16, "max_seq_len": 4096, "write_seq_len": 32, "head_dim": 256, "position_mode": "per_batch", "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # decode shared positions
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 2048, "write_seq_len": 1, "head_dim": 128, "position_mode": "shared", "dtype": "FP32"},
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 2048, "write_seq_len": 1, "head_dim": 128, "position_mode": "shared", "dtype": "FP16"},
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 2048, "write_seq_len": 1, "head_dim": 128, "position_mode": "shared", "dtype": "BF16"},

        # decode per-batch positions
        {"batch_size": 8, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 1, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP32"},
        {"batch_size": 8, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 1, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP16"},
        {"batch_size": 8, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 1, "head_dim": 128, "position_mode": "per_batch", "dtype": "BF16"},

        # prefill chunk writes
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 256, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP32"},
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 256, "head_dim": 128, "position_mode": "per_batch", "dtype": "FP16"},
        {"batch_size": 4, "num_kv_heads": 8, "max_seq_len": 4096, "write_seq_len": 256, "head_dim": 128, "position_mode": "per_batch", "dtype": "BF16"},

        # wider head dimension
        {"batch_size": 2, "num_kv_heads": 16, "max_seq_len": 4096, "write_seq_len": 32, "head_dim": 256, "position_mode": "per_batch", "dtype": "FP32"},
        {"batch_size": 2, "num_kv_heads": 16, "max_seq_len": 4096, "write_seq_len": 32, "head_dim": 256, "position_mode": "per_batch", "dtype": "FP16"},
        {"batch_size": 2, "num_kv_heads": 16, "max_seq_len": 4096, "write_seq_len": 32, "head_dim": 256, "position_mode": "per_batch", "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "num_kv_heads": [1, 4, 8, 16],
    "max_seq_len": [512, 1024, 2048, 4096, 8192],
    "write_seq_len": [1, 8, 16, 32, 128, 256, 512],
    "head_dim": [64, 80, 96, 128, 256],
    "position_mode": ["shared", "per_batch"],
    "dtype": ["FP32", "FP16", "BF16"],
}
