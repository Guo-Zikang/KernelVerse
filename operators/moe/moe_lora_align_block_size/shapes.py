"""Shape templates for moe_lora_align_block_size benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 512, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 2048, "max_num_m_blocks": 16, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 8192, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 32768, "max_num_m_blocks": 256, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},

        # LoRA adapter count sweep
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 2, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 32, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},

        # top-k and block-size sweep
        {"num_tokens": 2048, "topk": 1, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 2048, "max_num_m_blocks": 16, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 8, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 16384, "max_num_m_blocks": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 4096, "topk": 4, "num_experts": 128, "block_size": 64, "max_loras": 16, "max_num_tokens_padded": 16384, "max_num_m_blocks": 256, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
    ],
    "dtype_sweep": [
        # INT32 input/output
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT64", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 128, "max_loras": 8, "max_num_tokens_padded": 8192, "max_num_m_blocks": 64, "input_index_dtype": "INT32", "output_index_dtype": "INT64"},

        # larger routing fanout
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 128, "max_loras": 16, "max_num_tokens_padded": 32768, "max_num_m_blocks": 256, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 128, "max_loras": 16, "max_num_tokens_padded": 32768, "max_num_m_blocks": 256, "input_index_dtype": "INT64", "output_index_dtype": "INT32"},
        {"num_tokens": 4096, "topk": 8, "num_experts": 128, "block_size": 128, "max_loras": 16, "max_num_tokens_padded": 32768, "max_num_m_blocks": 256, "input_index_dtype": "INT32", "output_index_dtype": "INT64"},

        # many LoRA adapters
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 64, "max_loras": 32, "max_num_tokens_padded": 8192, "max_num_m_blocks": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 64, "max_loras": 32, "max_num_tokens_padded": 8192, "max_num_m_blocks": 128, "input_index_dtype": "INT64", "output_index_dtype": "INT32"},
        {"num_tokens": 2048, "topk": 4, "num_experts": 64, "block_size": 64, "max_loras": 32, "max_num_tokens_padded": 8192, "max_num_m_blocks": 128, "input_index_dtype": "INT32", "output_index_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [128, 512, 1024, 2048, 4096, 8192],
    "topk": [1, 2, 4, 8],
    "num_experts": [16, 32, 64, 128],
    "block_size": [16, 32, 64, 128, 256],
    "max_loras": [2, 4, 8, 16, 32],
    "max_num_tokens_padded": [1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "max_num_m_blocks": [16, 32, 64, 128, 256, 512, 1024],
    "input_index_dtype": ["INT32", "INT64"],
    "output_index_dtype": ["INT32", "INT64"],
}
