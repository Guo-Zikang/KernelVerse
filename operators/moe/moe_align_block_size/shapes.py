"""Shape templates for moe_align_block_size benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 128, "topk": 2, "num_experts": 32, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 2, "num_experts": 32, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 8192, "topk": 2, "num_experts": 32, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},

        # expert count sweep
        {"num_tokens": 2048, "topk": 4, "num_experts": 8, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 4, "num_experts": 256, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},

        # top-k and block-size sweep
        {"num_tokens": 2048, "topk": 1, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 8, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 4096, "topk": 4, "num_experts": 64, "block_size": 32, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 4096, "topk": 4, "num_experts": 64, "block_size": 256, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},

        # padded sorted token behavior
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": True},
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": True},
    ],
    "dtype_sweep": [
        # INT32 input/output
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT64", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT64", "pad_sorted_token_ids": False},

        # larger routing fanout
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 128, "input_index_dtype": "INT64", "output_index_dtype": "INT32", "pad_sorted_token_ids": False},
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT64", "pad_sorted_token_ids": False},

        # padded token ids enabled
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32", "pad_sorted_token_ids": True},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT64", "output_index_dtype": "INT32", "pad_sorted_token_ids": True},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT64", "pad_sorted_token_ids": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [128, 512, 1024, 2048, 4096, 8192],
    "topk": [1, 2, 4, 8],
    "num_experts": [8, 16, 32, 64, 128, 256],
    "block_size": [16, 32, 64, 128, 256],
    "input_index_dtype": ["INT32", "INT64"],
    "output_index_dtype": ["INT32", "INT64"],
    "pad_sorted_token_ids": [False, True],
}
