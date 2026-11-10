"""Shape templates for sparse_decode_topk benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_rows": 128, "num_cols": 4096, "topk": 16, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # row count sweep
        {"num_rows": 1, "num_cols": 4096, "topk": 16, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "topk": 16, "dtype": "FP32"},
        {"num_rows": 512, "num_cols": 4096, "topk": 16, "dtype": "FP32"},

        # cached sequence length sweep
        {"num_rows": 128, "num_cols": 256, "topk": 8, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 2048, "topk": 16, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 32000, "topk": 64, "dtype": "FP32"},

        # top-k sweep
        {"num_rows": 128, "num_cols": 4096, "topk": 1, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "topk": 32, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "topk": 100, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard decode index selection
        {"num_rows": 128, "num_cols": 4096, "topk": 16, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "topk": 16, "dtype": "FP16"},
        {"num_rows": 128, "num_cols": 4096, "topk": 16, "dtype": "BF16"},

        # long cache
        {"num_rows": 64, "num_cols": 32000, "topk": 64, "dtype": "FP32"},
        {"num_rows": 64, "num_cols": 32000, "topk": 64, "dtype": "FP16"},
        {"num_rows": 64, "num_cols": 32000, "topk": 64, "dtype": "BF16"},

        # many rows
        {"num_rows": 512, "num_cols": 2048, "topk": 8, "dtype": "FP32"},
        {"num_rows": 512, "num_cols": 2048, "topk": 8, "dtype": "FP16"},
        {"num_rows": 512, "num_cols": 2048, "topk": 8, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_rows": [1, 8, 32, 64, 128, 256, 512, 1024],
    "num_cols": [8, 16, 64, 256, 1024, 2048, 4096, 8192, 32000],
    "topk": [1, 2, 4, 8, 16, 32, 64, 100],
    "dtype": ["FP32", "FP16", "BF16"],
}
