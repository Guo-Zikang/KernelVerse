"""Shape templates for grouped_moe_mm benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_experts": 8, "tokens_per_expert": 16, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 32, "tokens_per_expert": 512, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},

        # expert-count sweep
        {"num_experts": 4, "tokens_per_expert": 256, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 16, "tokens_per_expert": 256, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 64, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},

        # matrix-size sweep
        {"num_experts": 16, "tokens_per_expert": 128, "K": 2048, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 11008, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 8, "tokens_per_expert": 256, "K": 8192, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # per-token activation and per-channel weight scales
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "FP16"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": True, "dtype": "BF16"},

        # scalar activation scale and per-channel weight scales
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": False, "per_out_ch": True, "dtype": "FP32"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": False, "per_out_ch": True, "dtype": "FP16"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": False, "per_out_ch": True, "dtype": "BF16"},

        # per-token activation scale and scalar weight scale
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": False, "dtype": "FP32"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": False, "dtype": "FP16"},
        {"num_experts": 16, "tokens_per_expert": 128, "K": 4096, "N": 4096, "per_act_token": True, "per_out_ch": False, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_experts": [1, 2, 4, 8, 16, 32, 64],
    "tokens_per_expert": [1, 8, 16, 32, 128, 256, 512],
    "K": [512, 1024, 2048, 4096, 8192],
    "N": [512, 1024, 2048, 4096, 8192, 11008],
    "per_act_token": [True, False],
    "per_out_ch": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
