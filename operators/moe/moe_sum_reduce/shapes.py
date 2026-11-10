"""Shape templates for moe_sum_reduce benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 512, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 8192, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},

        # hidden size sweep
        {"num_tokens": 2048, "topk": 4, "hidden_size": 1024, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 2048, "topk": 4, "hidden_size": 8192, "dtype": "FP32", "routed_scaling_factor": 2.5},

        # top-k and scaling sweep
        {"num_tokens": 2048, "topk": 1, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 1.0},
        {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 4096, "topk": 8, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 0.5},
    ],
    "dtype_sweep": [
        # standard routed reduce
        {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "FP16", "routed_scaling_factor": 2.5},
        {"num_tokens": 2048, "topk": 4, "hidden_size": 4096, "dtype": "BF16", "routed_scaling_factor": 2.5},

        # larger token batch
        {"num_tokens": 8192, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 2.5},
        {"num_tokens": 8192, "topk": 4, "hidden_size": 4096, "dtype": "FP16", "routed_scaling_factor": 2.5},
        {"num_tokens": 8192, "topk": 4, "hidden_size": 4096, "dtype": "BF16", "routed_scaling_factor": 2.5},

        # high top-k reduce
        {"num_tokens": 4096, "topk": 8, "hidden_size": 4096, "dtype": "FP32", "routed_scaling_factor": 1.0},
        {"num_tokens": 4096, "topk": 8, "hidden_size": 4096, "dtype": "FP16", "routed_scaling_factor": 1.0},
        {"num_tokens": 4096, "topk": 8, "hidden_size": 4096, "dtype": "BF16", "routed_scaling_factor": 1.0},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [128, 512, 1024, 2048, 4096, 8192, 16384],
    "topk": [1, 2, 4, 8],
    "hidden_size": [512, 1024, 2048, 4096, 8192, 11008],
    "dtype": ["FP32", "FP16", "BF16"],
    "routed_scaling_factor": [0.5, 1.0, 2.5],
}
