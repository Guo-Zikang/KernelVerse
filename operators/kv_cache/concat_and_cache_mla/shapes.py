"""Shape templates for concat_and_cache_mla benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 128, "num_blocks": 16, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 8192, "num_blocks": 256, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},

        # MLA dimension sweep
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 256, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 128, "dtype": "FP32"},
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "dtype": "FP32"},

        # cache layout sweep
        {"num_tokens": 1024, "num_blocks": 32, "block_size": 64, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 4096, "num_blocks": 128, "block_size": 32, "latent_dim": 768, "rope_dim": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard MLA cache update
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "FP16"},
        {"num_tokens": 1024, "num_blocks": 64, "block_size": 32, "latent_dim": 512, "rope_dim": 64, "dtype": "BF16"},

        # wider latent cache entry
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "dtype": "FP32"},
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "dtype": "FP16"},
        {"num_tokens": 2048, "num_blocks": 128, "block_size": 32, "latent_dim": 1024, "rope_dim": 128, "dtype": "BF16"},

        # smaller cache block layout
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "dtype": "FP32"},
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "dtype": "FP16"},
        {"num_tokens": 1024, "num_blocks": 128, "block_size": 16, "latent_dim": 512, "rope_dim": 64, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 512, 1024, 2048, 4096, 8192],
    "num_blocks": [1, 4, 16, 32, 64, 128, 256, 512],
    "block_size": [16, 32, 64, 128],
    "latent_dim": [128, 256, 512, 768, 1024],
    "rope_dim": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
