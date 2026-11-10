"""Shape templates for mm_encoder_attention_varlen benchmark cases."""

SINGLE_SHAPES = {
    "default": {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # short and moderately ragged multimodal batches
        {"seq_lens": [128, 96, 64, 32], "num_heads": 8, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [256, 192, 128, 64], "num_heads": 8, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 64, "dtype": "FP32"},

        # common longer packed encoder layouts
        {"seq_lens": [1024, 768, 512, 256], "num_heads": 16, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [2048, 1536, 1024, 512], "num_heads": 16, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [4096, 3072, 2048, 1024], "num_heads": 16, "head_dim": 64, "dtype": "FP32"},

        # head count and head dimension variation
        {"seq_lens": [512, 384, 256, 128], "num_heads": 4, "head_dim": 128, "dtype": "FP32"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 16, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 32, "head_dim": 64, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # moderate packed batch
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 64, "dtype": "FP16"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 64, "dtype": "BF16"},

        # long packed batch
        {"seq_lens": [1024, 768, 512, 256], "num_heads": 16, "head_dim": 64, "dtype": "FP32"},
        {"seq_lens": [1024, 768, 512, 256], "num_heads": 16, "head_dim": 64, "dtype": "FP16"},
        {"seq_lens": [1024, 768, 512, 256], "num_heads": 16, "head_dim": 64, "dtype": "BF16"},

        # wider head dimension
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 128, "dtype": "FP32"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 128, "dtype": "FP16"},
        {"seq_lens": [512, 384, 256, 128], "num_heads": 8, "head_dim": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "seq_lens": [
        [64, 48, 32, 16],
        [128, 96, 64, 32],
        [256, 192, 128, 64],
        [512, 384, 256, 128],
        [768, 512, 384, 256],
        [1024, 768, 512, 256],
        [1536, 1024, 768, 512],
        [2048, 1536, 1024, 512],
        [4096, 3072, 2048, 1024],
        [8192, 4096, 2048, 1024],
        [1024],
        [2048],
        [4096],
        [8192],
        [128, 127, 63, 31],
    ],
    "num_heads": [4, 8, 16, 32],
    "head_dim": [32, 64, 80, 96, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
