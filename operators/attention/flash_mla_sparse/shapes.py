"""Shape templates for flash_mla_sparse benchmark cases."""

SINGLE_SHAPES = {
    "default": {"seq_q": 128, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # query length sweep
        {"seq_q": 32, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 512, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},

        # KV length sweep
        {"seq_q": 128, "seq_kv": 1024, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 8192, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},

        # selected candidate count sweep
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 16, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 128, "dtype": "FP32"},

        # head count and head dimension sweep
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 8, "head_dim": 128, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 32, "head_dim": 64, "topk": 64, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # moderate sparse attention
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "FP16"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 16, "head_dim": 64, "topk": 64, "dtype": "BF16"},

        # long KV with larger topk
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 128, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 128, "dtype": "FP16"},
        {"seq_q": 128, "seq_kv": 4096, "num_heads": 16, "head_dim": 64, "topk": 128, "dtype": "BF16"},

        # wider head dimension
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 8, "head_dim": 128, "topk": 64, "dtype": "FP32"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 8, "head_dim": 128, "topk": 64, "dtype": "FP16"},
        {"seq_q": 128, "seq_kv": 2048, "num_heads": 8, "head_dim": 128, "topk": 64, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "seq_q": [16, 32, 64, 128, 256, 512],
    "seq_kv": [512, 1024, 2048, 4096, 8192],
    "num_heads": [4, 8, 16, 32],
    "head_dim": [32, 64, 80, 96, 128],
    "topk": [8, 16, 32, 64, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
}
