"""Shape templates for batched_prefill_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # full prefill prompt length sweep
        {"q_lens": [128, 96, 64, 32], "kv_lens": [128, 96, 64, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [2048, 1536, 1024, 512], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # highly ragged serving batches
        {"q_lens": [1024, 128, 64, 32, 16, 8], "kv_lens": [1024, 128, 64, 32, 16, 8], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [4096, 1024, 256, 64], "kv_lens": [4096, 1024, 256, 64], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [8192, 512, 128, 32], "kv_lens": [8192, 512, 128, 32], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # chunked prefill with existing KV context
        {"q_lens": [128, 128, 128, 128], "kv_lens": [1024, 768, 512, 256], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [256, 256, 256, 256], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [512, 512, 512, 512], "kv_lens": [4096, 3072, 2048, 1024], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # MHA/GQA/MQA head mapping sweep
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 16, "num_kv_heads": 16, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard GQA full prefill
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"q_lens": [512, 384, 256, 128], "kv_lens": [512, 384, 256, 128], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "BF16"},

        # long ragged prompt batch
        {"q_lens": [2048, 1536, 1024, 512], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [2048, 1536, 1024, 512], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"q_lens": [2048, 1536, 1024, 512], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "BF16"},

        # chunked prefill with existing KV context
        {"q_lens": [256, 256, 256, 256], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP32"},
        {"q_lens": [256, 256, 256, 256], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP16"},
        {"q_lens": [256, 256, 256, 256], "kv_lens": [2048, 1536, 1024, 512], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "q_lens": [
        [64, 48, 32, 16],
        [128, 96, 64, 32],
        [256, 192, 128, 64],
        [512, 384, 256, 128],
        [1024, 768, 512, 256],
        [2048, 1536, 1024, 512],
        [4096, 1024, 256, 64],
        [8192, 512, 128, 32],
        [128, 128, 128, 128],
        [256, 256, 256, 256],
        [512],
        [2048],
    ],
    "kv_lens": [
        [64, 48, 32, 16],
        [128, 96, 64, 32],
        [256, 192, 128, 64],
        [512, 384, 256, 128],
        [1024, 768, 512, 256],
        [2048, 1536, 1024, 512],
        [4096, 1024, 256, 64],
        [8192, 512, 128, 32],
        [1024, 768, 512, 256],
        [2048, 1536, 1024, 512],
        [4096],
        [8192],
    ],
    "num_q_heads": [8, 16, 32, 40],
    "num_kv_heads": [1, 2, 4, 8, 16],
    "head_dim": [64, 80, 96, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
