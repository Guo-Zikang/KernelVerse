"""Shape templates for concat_and_cache_mla_rope benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 128, "num_blocks": 16, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 8192, "num_blocks": 256, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 8192, "dtype": "FP32", "is_neox": True},

        # MLA dimension sweep
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 256, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 128, "max_position": 4096, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "max_position": 8192, "dtype": "FP32", "is_neox": True},

        # cache layout and rotary style sweep
        {"num_tokens": 1024, "num_blocks": 32, "block_size": 64, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": False},
        {"num_tokens": 4096, "num_blocks": 128, "block_size": 32, "latent_dim": 768, "rope_dim": 128, "max_position": 16384, "dtype": "FP32", "is_neox": False},
    ],
    "dtype_sweep": [
        # standard MLA cache update
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP16", "is_neox": True},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "BF16", "is_neox": True},

        # wider latent cache entry
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "max_position": 8192, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "max_position": 8192, "dtype": "FP16", "is_neox": True},
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "max_position": 8192, "dtype": "BF16", "is_neox": True},

        # interleaved rotary layout
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP32", "is_neox": False},
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "FP16", "is_neox": False},
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "max_position": 4096, "dtype": "BF16", "is_neox": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 512, 1024, 2048, 4096, 8192],
    "num_blocks": [1, 4, 16, 32, 64, 128, 256, 512],
    "block_size": [16, 32, 64, 128],
    "latent_dim": [128, 256, 512, 768, 1024],
    "rope_dim": [32, 64, 128],
    "max_position": [1024, 2048, 4096, 8192, 16384],
    "dtype": ["FP32", "FP16", "BF16"],
    "is_neox": [True, False],
}
