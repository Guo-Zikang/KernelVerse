"""Shape templates for sparse_prefill_topk benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # row count sweep
        {"num_rows": 1, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "FP32"},
        {"num_rows": 512, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "FP32"},

        # prefill candidate range sweep
        {"num_rows": 128, "num_cols": 4096, "range_len": 128, "topk": 8, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 8192, "range_len": 2048, "topk": 32, "dtype": "FP32"},
        {"num_rows": 64, "num_cols": 32000, "range_len": 8192, "topk": 64, "dtype": "FP32"},

        # top-k sweep
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 1, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 32, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 100, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard prefill index selection
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "FP32"},
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "FP16"},
        {"num_rows": 128, "num_cols": 4096, "range_len": 1024, "topk": 16, "dtype": "BF16"},

        # long prefill candidate range
        {"num_rows": 64, "num_cols": 32000, "range_len": 8192, "topk": 64, "dtype": "FP32"},
        {"num_rows": 64, "num_cols": 32000, "range_len": 8192, "topk": 64, "dtype": "FP16"},
        {"num_rows": 64, "num_cols": 32000, "range_len": 8192, "topk": 64, "dtype": "BF16"},

        # many rows
        {"num_rows": 512, "num_cols": 2048, "range_len": 512, "topk": 8, "dtype": "FP32"},
        {"num_rows": 512, "num_cols": 2048, "range_len": 512, "topk": 8, "dtype": "FP16"},
        {"num_rows": 512, "num_cols": 2048, "range_len": 512, "topk": 8, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_rows": [1, 8, 32, 64, 128, 256, 512, 1024],
    "num_cols": [64, 256, 1024, 2048, 4096, 8192, 32000],
    "range_len": [8, 16, 64, 128, 512, 1024, 2048, 8192],
    "topk": [1, 2, 4, 8, 16, 32, 64, 100],
    "dtype": ["FP32", "FP16", "BF16"],
}
