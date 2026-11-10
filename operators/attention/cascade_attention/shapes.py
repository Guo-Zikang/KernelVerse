"""Shape templates for cascade_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # shared prefix length sweep
        {"batch_size": 8, "shared_len": 128, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 1024, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 8192, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # unique suffix raggedness sweep
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [64, 64, 64, 64, 64, 64, 64, 64], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [2048, 512, 128, 32, 16, 8, 4, 0], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # batch size sweep
        {"batch_size": 1, "shared_len": 4096, "unique_lens": [512], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 16, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16, 512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 32, "shared_len": 2048, "unique_lens": [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 0, 0, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 0, 0, 128, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},

        # MHA/GQA/MQA head mapping sweep
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 16, "num_kv_heads": 16, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard shared-prefix cascade
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"batch_size": 8, "shared_len": 2048, "unique_lens": [512, 384, 256, 128, 96, 64, 32, 16], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "BF16"},

        # long shared prefix
        {"batch_size": 4, "shared_len": 8192, "unique_lens": [512, 128, 32, 0], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 4, "shared_len": 8192, "unique_lens": [512, 128, 32, 0], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"batch_size": 4, "shared_len": 8192, "unique_lens": [512, 128, 32, 0], "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "BF16"},

        # MQA cascade
        {"batch_size": 8, "shared_len": 4096, "unique_lens": [1024, 512, 256, 128, 64, 32, 16, 8], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "shared_len": 4096, "unique_lens": [1024, 512, 256, 128, 64, 32, 16, 8], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "FP16"},
        {"batch_size": 8, "shared_len": 4096, "unique_lens": [1024, 512, 256, 128, 64, 32, 16, 8], "num_q_heads": 32, "num_kv_heads": 1, "head_dim": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 4, 8, 16, 32],
    "shared_len": [128, 512, 1024, 2048, 4096, 8192],
    "unique_lens": [
        [64],
        [512],
        [64, 64, 64, 64],
        [512, 384, 256, 128],
        [1024, 512, 128, 0],
        [512, 384, 256, 128, 96, 64, 32, 16],
        [2048, 512, 128, 32, 16, 8, 4, 0],
        [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 0, 0, 128, 64],
        [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 0, 0, 128, 64, 32, 16, 8, 4, 0, 0, 256, 128, 64, 32, 16, 8, 4, 0, 0, 0],
    ],
    "num_q_heads": [8, 16, 32, 40],
    "num_kv_heads": [1, 2, 4, 8, 16],
    "head_dim": [64, 80, 96, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
