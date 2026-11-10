"""Shape templates for flash_mla_with_kvcache benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode KV length sweep
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 512, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 1024, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 4096, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 8192, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},

        # query token count sweep
        {"batch_size": 2, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 2, "seq_len_q": 8, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 2, "seq_len_q": 16, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 2, "seq_len_q": 32, "seq_len_k": 4096, "num_heads": 16, "head_dim": 128, "head_dim_v": 64, "cache_pattern": "full", "dtype": "FP32"},

        # value dimension sweep
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 64, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 128, "head_dim_v": 64, "cache_pattern": "full", "dtype": "FP32"},

        # ragged effective cache lengths
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 8, "seq_len_q": 1, "seq_len_k": 4096, "num_heads": 32, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 16, "seq_len_k": 4096, "num_heads": 16, "head_dim": 128, "head_dim_v": 64, "cache_pattern": "ragged", "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # decode MLA cache case
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP16"},
        {"batch_size": 4, "seq_len_q": 1, "seq_len_k": 2048, "num_heads": 16, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "BF16"},

        # long KV cache
        {"batch_size": 8, "seq_len_q": 1, "seq_len_k": 4096, "num_heads": 32, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP32"},
        {"batch_size": 8, "seq_len_q": 1, "seq_len_k": 4096, "num_heads": 32, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "FP16"},
        {"batch_size": 8, "seq_len_q": 1, "seq_len_k": 4096, "num_heads": 32, "head_dim": 64, "head_dim_v": 128, "cache_pattern": "full", "dtype": "BF16"},

        # ragged cache with different value dimension
        {"batch_size": 4, "seq_len_q": 16, "seq_len_k": 4096, "num_heads": 16, "head_dim": 128, "head_dim_v": 64, "cache_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 4, "seq_len_q": 16, "seq_len_k": 4096, "num_heads": 16, "head_dim": 128, "head_dim_v": 64, "cache_pattern": "ragged", "dtype": "FP16"},
        {"batch_size": 4, "seq_len_q": 16, "seq_len_k": 4096, "num_heads": 16, "head_dim": 128, "head_dim_v": 64, "cache_pattern": "ragged", "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "seq_len_q": [1, 8, 16, 32, 64],
    "seq_len_k": [512, 1024, 2048, 4096, 8192],
    "num_heads": [8, 16, 32],
    "head_dim": [64, 128],
    "head_dim_v": [64, 128, 256],
    "cache_pattern": ["full", "ragged"],
    "dtype": ["FP32", "FP16", "BF16"],
}
