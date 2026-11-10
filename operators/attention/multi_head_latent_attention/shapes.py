"""Shape templates for multi_head_latent_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 4,
        "query_seq_len": 1,
        "num_q_heads": 16,
        "qk_head_dim": 576,
        "v_head_dim": 512,
        "num_blocks": 256,
        "block_size": 16,
        "max_blocks_per_seq": 64,
        "cache_seq_len": 512,
        "causal": True,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode cache length sweep
        {"batch_size": 1, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 64, "block_size": 16, "max_blocks_per_seq": 32, "cache_seq_len": 256, "causal": True, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 256, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 8, "query_seq_len": 1, "num_q_heads": 32, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 1024, "block_size": 16, "max_blocks_per_seq": 128, "cache_seq_len": 1024, "causal": True, "dtype": "FP32"},

        # query token count sweep
        {"batch_size": 2, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 16, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 32, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 64, "num_q_heads": 32, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 512, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},

        # latent dimension variants
        {"batch_size": 2, "query_seq_len": 1, "num_q_heads": 8, "qk_head_dim": 256, "v_head_dim": 256, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 1, "num_q_heads": 32, "qk_head_dim": 768, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},

        # block size and causal mode sweep
        {"batch_size": 2, "query_seq_len": 16, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 64, "block_size": 32, "max_blocks_per_seq": 32, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 16, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 64, "block_size": 32, "max_blocks_per_seq": 32, "cache_seq_len": 512, "causal": False, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # decode case
        {"batch_size": 4, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 256, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 4, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 256, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP16"},
        {"batch_size": 4, "query_seq_len": 1, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 256, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "BF16"},

        # dense decode
        {"batch_size": 2, "query_seq_len": 32, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP32"},
        {"batch_size": 2, "query_seq_len": 32, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "FP16"},
        {"batch_size": 2, "query_seq_len": 32, "num_q_heads": 16, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 128, "block_size": 16, "max_blocks_per_seq": 64, "cache_seq_len": 512, "causal": True, "dtype": "BF16"},

        # large head count
        {"batch_size": 8, "query_seq_len": 1, "num_q_heads": 32, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 1024, "block_size": 16, "max_blocks_per_seq": 128, "cache_seq_len": 1024, "causal": True, "dtype": "FP32"},
        {"batch_size": 8, "query_seq_len": 1, "num_q_heads": 32, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 1024, "block_size": 16, "max_blocks_per_seq": 128, "cache_seq_len": 1024, "causal": True, "dtype": "FP16"},
        {"batch_size": 8, "query_seq_len": 1, "num_q_heads": 32, "qk_head_dim": 576, "v_head_dim": 512, "num_blocks": 1024, "block_size": 16, "max_blocks_per_seq": 128, "cache_seq_len": 1024, "causal": True, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "query_seq_len": [1, 16, 32, 64],
    "num_q_heads": [8, 16, 32],
    "qk_head_dim": [256, 576, 768],
    "v_head_dim": [256, 512],
    "num_blocks": [64, 128, 256, 512, 1024],
    "block_size": [16, 32],
    "max_blocks_per_seq": [32, 64, 128],
    "cache_seq_len": [256, 512, 1024],
    "causal": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
