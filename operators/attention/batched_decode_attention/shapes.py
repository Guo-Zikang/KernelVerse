"""Shape templates for batched_decode_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # KV cache length sweep
        {"batch_size": 16, "max_seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "FP32"},
        {"batch_size": 16, "max_seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "FP32"},
        {"batch_size": 16, "max_seq_len": 8192, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "FP32"},

        # batch size sweep
        {"batch_size": 1, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "FP32"},
        {"batch_size": 32, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 128, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "highly_ragged", "dtype": "FP32"},

        # MHA/GQA/MQA head mapping sweep
        {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 16, "num_kv_heads": 16, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},

        # head dimension and raggedness sweep
        {"batch_size": 32, "max_seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64, "seq_len_pattern": "uniform_medium", "dtype": "FP32"},
        {"batch_size": 32, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 32, "max_seq_len": 8192, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "highly_ragged", "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard GQA decode
        {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP32"},
        {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "FP16"},
        {"batch_size": 16, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "ragged", "dtype": "BF16"},

        # long KV cache
        {"batch_size": 8, "max_seq_len": 8192, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "FP32"},
        {"batch_size": 8, "max_seq_len": 8192, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "FP16"},
        {"batch_size": 8, "max_seq_len": 8192, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len_pattern": "uniform_long", "dtype": "BF16"},

        # large ragged batch
        {"batch_size": 64, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "seq_len_pattern": "highly_ragged", "dtype": "FP32"},
        {"batch_size": 64, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "seq_len_pattern": "highly_ragged", "dtype": "FP16"},
        {"batch_size": 64, "max_seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "seq_len_pattern": "highly_ragged", "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 4, 8, 16, 32, 64, 128],
    "max_seq_len": [512, 1024, 2048, 4096, 8192, 16384, 32768],
    "num_q_heads": [8, 16, 32, 40],
    "num_kv_heads": [1, 2, 4, 8, 16],
    "head_dim": [64, 128],
    "seq_len_pattern": ["uniform_short", "uniform_medium", "uniform_long", "ragged", "highly_ragged"],
    "dtype": ["FP32", "FP16", "BF16"],
}
