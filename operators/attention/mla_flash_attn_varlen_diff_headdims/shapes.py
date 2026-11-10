"""Shape templates for mla_flash_attn_varlen_diff_headdims benchmark cases."""

SINGLE_SHAPES = {
    "default": {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # variable-length sequence pattern sweep
        {"q_seq_lens": [128, 96, 64, 32], "kv_seq_lens": [192, 128, 96, 64], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [1024, 768, 512, 256], "kv_seq_lens": [1536, 1024, 768, 512], "num_heads": 16, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},

        # QK and V head dimension sweep
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 64, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 128, "v_head_dim": 64, "dtype": "FP32", "causal": True},

        # head count and causal mode sweep
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 4, "qk_head_dim": 128, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 16, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": False},
    ],
    "dtype_sweep": [
        # moderate varlen MLA attention
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP16", "causal": True},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "BF16", "causal": True},

        # long packed batch
        {"q_seq_lens": [1024, 768, 512, 256], "kv_seq_lens": [1536, 1024, 768, 512], "num_heads": 16, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP32", "causal": True},
        {"q_seq_lens": [1024, 768, 512, 256], "kv_seq_lens": [1536, 1024, 768, 512], "num_heads": 16, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "FP16", "causal": True},
        {"q_seq_lens": [1024, 768, 512, 256], "kv_seq_lens": [1536, 1024, 768, 512], "num_heads": 16, "qk_head_dim": 64, "v_head_dim": 128, "dtype": "BF16", "causal": True},

        # different QK and V head dimensions
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 128, "v_head_dim": 64, "dtype": "FP32", "causal": False},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 128, "v_head_dim": 64, "dtype": "FP16", "causal": False},
        {"q_seq_lens": [512, 384, 256, 128], "kv_seq_lens": [768, 512, 384, 256], "num_heads": 8, "qk_head_dim": 128, "v_head_dim": 64, "dtype": "BF16", "causal": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "q_seq_lens": [
        [64, 48, 32, 16],
        [128, 96, 64, 32],
        [256, 192, 128, 64],
        [512, 384, 256, 128],
        [1024, 768, 512, 256],
        [1024],
        [2048],
        [128, 127, 63, 31],
    ],
    "kv_seq_lens": [
        [96, 64, 48, 32],
        [192, 128, 96, 64],
        [384, 256, 192, 128],
        [768, 512, 384, 256],
        [1536, 1024, 768, 512],
        [1024],
        [2048],
        [192, 191, 95, 63],
    ],
    "num_heads": [4, 8, 16, 32],
    "qk_head_dim": [32, 64, 96, 128],
    "v_head_dim": [32, 64, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
    "causal": [True, False],
}
