"""Shape templates for linformer_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 2,
        "seq_len": 512,
        "num_heads": 8,
        "head_dim": 64,
        "projected_seq_len": 128,
        "share_kv_projection": False,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # context length sweep
        {"batch_size": 1, "seq_len": 256, "num_heads": 8, "head_dim": 64, "projected_seq_len": 64, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 8, "head_dim": 64, "projected_seq_len": 256, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "projected_seq_len": 256, "share_kv_projection": False, "dtype": "FP32"},

        # projected sequence length sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "projected_seq_len": 32, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "projected_seq_len": 64, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "projected_seq_len": 256, "share_kv_projection": False, "dtype": "FP32"},

        # hidden size and batch sweep
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 32, "head_dim": 128, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 40, "head_dim": 128, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},

        # K/V projection sharing sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": True, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "projected_seq_len": 256, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "projected_seq_len": 256, "share_kv_projection": False, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "projected_seq_len": 256, "share_kv_projection": False, "dtype": "BF16"},

        # large hidden
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 32, "head_dim": 128, "projected_seq_len": 128, "share_kv_projection": False, "dtype": "BF16"},

        # shared projection path
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": True, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "projected_seq_len": 128, "share_kv_projection": True, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4],
    "seq_len": [128, 256, 512, 1024, 2048],
    "num_heads": [8, 16, 32, 40],
    "head_dim": [64, 96, 128],
    "projected_seq_len": [32, 64, 128, 256, 512],
    "share_kv_projection": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
