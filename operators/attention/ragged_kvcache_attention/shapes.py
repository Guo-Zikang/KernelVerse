"""Shape templates for ragged_kvcache_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # KV length sweep
        {"kv_lens": [128, 96, 64, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [1024, 768, 512, 256], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [4096, 3072, 2048, 1024], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # highly ragged serving batches
        {"kv_lens": [1024, 128, 64, 32, 16, 8], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [4096, 1024, 256, 64], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [8192, 512, 128, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # batch size sweep
        {"kv_lens": [4096], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [2048, 1536, 1024, 512, 384, 256, 128, 64], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [1024, 896, 768, 640, 512, 384, 256, 128, 96, 64, 48, 32, 24, 16, 12, 8], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # MHA/GQA/MQA head mapping sweep
        {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 16, "num_kv_heads": 16, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard GQA ragged decode
        {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "BF16"},

        # long highly ragged batch
        {"kv_lens": [8192, 512, 128, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [8192, 512, 128, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"kv_lens": [8192, 512, 128, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "BF16"},

        # MQA ragged decode
        {"kv_lens": [4096, 1024, 256, 64], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP32"},
        {"kv_lens": [4096, 1024, 256, 64], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP16"},
        {"kv_lens": [4096, 1024, 256, 64], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "kv_lens": [
        [64, 48, 32, 16],
        [128, 96, 64, 32],
        [256, 192, 128, 64],
        [512, 384, 256, 128],
        [1024, 768, 512, 256],
        [2048, 1536, 1024, 512],
        [4096, 3072, 2048, 1024],
        [8192, 4096, 2048, 1024],
        [1024, 128, 64, 32, 16, 8],
        [4096, 1024, 256, 64],
        [8192, 512, 128, 32],
        [4096],
        [8192],
    ],
    "num_q_heads": [8, 16, 32, 40],
    "num_kv_heads": [1, 2, 4, 8, 16],
    "head_dim": [64, 80, 96, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
