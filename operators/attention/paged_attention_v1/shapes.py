"""Shape templates for paged_attention_v1 benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 8,
        "seq_len": 2048,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "input_dtype": "FP32",
        "output_dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode sequence length sweep
        {"batch_size": 4, "seq_len": 512, "num_q_heads": 16, "num_kv_heads": 4, "head_dim": 64, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_q_heads": 16, "num_kv_heads": 4, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 8, "seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 32, "input_dtype": "FP32", "output_dtype": "FP32"},

        # batch size sweep
        {"batch_size": 1, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 4, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 16, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},

        # grouped-query ratio sweep
        {"batch_size": 4, "seq_len": 2048, "num_q_heads": 8, "num_kv_heads": 1, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 4, "seq_len": 2048, "num_q_heads": 16, "num_kv_heads": 4, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 4, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},

        # block size and head dimension sweep
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 32, "input_dtype": "FP32", "output_dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard decode
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "FP16", "output_dtype": "FP16"},
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "input_dtype": "BF16", "output_dtype": "BF16"},

        # long KV cache
        {"batch_size": 4, "seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 32, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 4, "seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 32, "input_dtype": "FP16", "output_dtype": "FP16"},
        {"batch_size": 4, "seq_len": 4096, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 32, "input_dtype": "BF16", "output_dtype": "BF16"},

        # multi-query style decode
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "block_size": 16, "input_dtype": "FP32", "output_dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "block_size": 16, "input_dtype": "FP16", "output_dtype": "FP16"},
        {"batch_size": 8, "seq_len": 2048, "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "block_size": 16, "input_dtype": "BF16", "output_dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "seq_len": [512, 1024, 2048, 4096],
    "num_q_heads": [8, 16, 32],
    "num_kv_heads": [1, 2, 4, 8],
    "head_dim": [64, 128],
    "block_size": [16, 32],
    "input_dtype": ["FP32", "FP16", "BF16"],
    "output_dtype": ["FP32", "FP16", "BF16"],
}
