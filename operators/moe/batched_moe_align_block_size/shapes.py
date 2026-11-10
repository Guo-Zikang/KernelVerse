"""Shape templates for batched_moe_align_block_size benchmark cases."""

SINGLE_SHAPES = {
    "default": {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # max token count per expert
        {"max_tokens_per_batch": 16, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 2048, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},

        # expert-count sweep
        {"max_tokens_per_batch": 256, "num_experts": 16, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 256, "num_experts": 128, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 128, "num_experts": 256, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},

        # block-size sweep
        {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 8, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 32, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 512, "num_experts": 64, "block_size": 64, "output_index_dtype": "INT32", "simulate_empty_batches": True},
    ],
    "dtype_sweep": [
        # standard output index dtype
        {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT64", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 256, "num_experts": 64, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": False},

        # larger expert count
        {"max_tokens_per_batch": 128, "num_experts": 256, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 128, "num_experts": 256, "block_size": 16, "output_index_dtype": "INT64", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 128, "num_experts": 256, "block_size": 16, "output_index_dtype": "INT32", "simulate_empty_batches": False},

        # larger block size
        {"max_tokens_per_batch": 512, "num_experts": 128, "block_size": 32, "output_index_dtype": "INT32", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 512, "num_experts": 128, "block_size": 32, "output_index_dtype": "INT64", "simulate_empty_batches": True},
        {"max_tokens_per_batch": 512, "num_experts": 128, "block_size": 32, "output_index_dtype": "INT32", "simulate_empty_batches": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "max_tokens_per_batch": [1, 8, 16, 32, 128, 256, 512, 2048],
    "num_experts": [1, 8, 16, 32, 64, 128, 256],
    "block_size": [1, 8, 16, 32, 64, 128],
    "output_index_dtype": ["INT32", "INT64"],
    "simulate_empty_batches": [True, False],
}
