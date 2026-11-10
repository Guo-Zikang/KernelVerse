"""Shape templates for flash_decoding_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 4,
        "query_seq_len": 1,
        "kv_seq_len": 2048,
        "num_heads": 16,
        "head_dim": 128,
        "chunk_size": 256,
        "causal": False,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode KV length sweep
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 512, "num_heads": 16, "head_dim": 128, "chunk_size": 128, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 1024, "num_heads": 16, "head_dim": 128, "chunk_size": 128, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 8192, "num_heads": 16, "head_dim": 128, "chunk_size": 512, "causal": False, "dtype": "FP32"},

        # query token count sweep
        {"batch_size": 2, "query_seq_len": 1, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 64, "chunk_size": 256, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 8, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 64, "chunk_size": 256, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 16, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 64, "chunk_size": 256, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 32, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": True, "dtype": "FP32"},

        # chunk size sweep
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 64, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 128, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 512, "causal": False, "dtype": "FP32"},

        # batch and head sweep
        {"batch_size": 1, "query_seq_len": 1, "kv_seq_len": 2048, "num_heads": 8, "head_dim": 64, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 8, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 32, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 16, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 32, "head_dim": 128, "chunk_size": 512, "causal": False, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # decode case
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP16"},
        {"batch_size": 4, "query_seq_len": 1, "kv_seq_len": 2048, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "BF16"},

        # long KV cache
        {"batch_size": 8, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 32, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP32"},
        {"batch_size": 8, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 32, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "FP16"},
        {"batch_size": 8, "query_seq_len": 1, "kv_seq_len": 4096, "num_heads": 32, "head_dim": 128, "chunk_size": 256, "causal": False, "dtype": "BF16"},

        # multi query tokens with causal mask
        {"batch_size": 2, "query_seq_len": 32, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 32, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": True, "dtype": "FP16"},
        {"batch_size": 2, "query_seq_len": 32, "kv_seq_len": 4096, "num_heads": 16, "head_dim": 128, "chunk_size": 256, "causal": True, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "query_seq_len": [1, 8, 16, 32, 64],
    "kv_seq_len": [512, 1024, 2048, 4096, 8192],
    "num_heads": [8, 16, 32],
    "head_dim": [64, 128],
    "chunk_size": [64, 128, 256, 512],
    "causal": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
