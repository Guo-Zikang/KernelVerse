# KernelVerse: Benchmarking LLMs for Inference Kernel Generation across Shapes, Precisions, and Accelerators

KernelVerse is a benchmark for evaluating LLM-generated inference kernels across variations in input shape, numerical precision, and accelerator hardware. It comprises 200 inference operators across 13 functional categories and supports five accelerators: NVIDIA A800, MetaX C500, Iluvatar BI-V150, Ascend 910B, and Biren 106B.

The framework provides four main workflows:

1. Benchmark the provided PyTorch reference operators.
2. Generate custom operator implementations with an OpenAI-compatible LLM API.
3. Compile or evaluate previously generated implementations.
4. Generate and evaluate kernels in a single workflow.

## Operator Suite

KernelVerse contains 200 inference operators curated from vLLM and SGLang, organized into 13 functional categories. Each task includes a PyTorch reference implementation, shape and precision configurations, and evaluation metadata. Representative operators are listed below.

| Category | Count | Examples |
| --- | ---: | --- |
| Activation | 9 | `gelu`, `silu`, `swiglu` |
| Attention | 53 | `batched_decode_attention`, `flash_attention`, `paged_attention_v1` |
| Convolution | 7 | `conv2d`, `conv3d`, `causal_conv1d_fwd` |
| GEMM | 8 | `matmul`, `addmm`, `bmm` |
| KV Cache | 13 | `kv_cache_write`, `concat_and_cache_mla`, `paged_kv_cache_update` |
| Logits Sampling | 16 | `softmax`, `top_k_renorm_probs`, `top_p_renorm_probs` |
| Mixture of Experts (MoE) | 19 | `grouped_moe_mm`, `moe_topk_routing`, `moe_permute` |
| Normalization | 6 | `layer_norm`, `rms_norm`, `add_rmsnorm` |
| Positional Embedding | 14 | `rotary_embedding`, `yarn_scaling_rotary_embedding`, `mrotary_embedding` |
| Quantization | 6 | `quantize_per_tensor`, `dequantize`, `dynamic_scaled_int8_quant` |
| Reduction | 8 | `sum_reduce`, `cumsum`, `logsumexp` |
| State Space | 18 | `selective_scan`, `selective_state_update`, `chunk_gated_delta_rule` |
| Tensor Operations | 23 | `gather`, `scatter`, `transpose` |
| **Total** | **200** | |

## Installation

Use a Linux environment with Python 3.9 or later. Install the matching driver, SDK, PyTorch runtime, and compiler toolchain for one backend below.

| Hardware | `--language` | PyTorch runtime and toolchain |
| --- | --- | --- |
| NVIDIA A800 | `cuda` | CUDA-enabled PyTorch, NVIDIA driver, and CUDA toolkit |
| MetaX C500 | `maca` | MetaX PyTorch build, MACA driver/SDK, and cuCC |
| Iluvatar BI-V150 | `tiangai` | Iluvatar PyTorch build, CoreX driver/SDK, and compiler toolchain |
| Ascend 910B | `ascendc` | Matching PyTorch and `torch_npu`, Ascend driver/firmware, CANN, and `msopgen` |
| Biren 106B | `biren` | Biren PyTorch runtime with `torch_br`, BIRENSUPA driver/SDK, and BRCC |

Activate your backend's Python environment and SDK settings, ensure the required dependencies are available, and verify that PyTorch can access your accelerator.

## Quick Start

Run the following examples on an NVIDIA A800 GPU from the repository root with the CUDA environment activated.

### 1. Benchmark a reference operator

Benchmark the PyTorch reference implementation of `addmm` in eager mode using its default configuration on CUDA device 0.

```bash
python main.py \
  --op addmm \
  --shape_mode single \
  --shape_name default \
  --language cuda \
  --mode eager \
  --device-ids 0
```

Results are printed to the console and saved under `results/`.

### 2. Generate and evaluate a kernel

Copy `.env.example` to `.env` and set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` for an OpenAI-compatible API. The following command generates and evaluates one `addmm` kernel on CUDA device 0:

```bash
python scripts/generate_and_verify.py \
  --op addmm \
  --shape_mode single \
  --language cuda \
  --device-ids 0 \
  --samples-per-op 1 \
  --llm-workers 1 \
  --run-dir runs/addmm_demo
```

Generation and evaluation results are saved to `generation.json` and `evaluation.json` under `runs/addmm_demo/`.

## Repository layout

```text
.
├── backends/                 # Accelerator-specific compilation/runtime backends
├── codegen/                  # Prompt construction and LLM code generation
├── evaluator/                # Generated-code compilation and evaluation
├── operators/                # Reference operators, shapes, and metadata
└── runner/                   # Benchmark, correctness, and profiling utilities
```

Each operator is stored under:

```text
operators/<category>/<operator>/
├── <operator>.py             # PyTorch reference Model and input generator
├── shapes.py                 # Named and family-based shape definitions
├── config.json               # Random-shape configuration
├── docs.md                   # Operator semantics and constraints
└── helpers.py                # Shape validation and input helpers
```

## Operator and shape selection

All main workflows accept one of the following selectors:

- `--op addmm` for one operator
- `--op addmm,bmm,conv2d` for several operators
- `--category attention` for one category
- `--all` for every registered operator

Three shape modes are supported:

- `single`: run one named shape from the operator's `shapes.py`.
- `family`: run one shape family defined in the operator's `shapes.py`, or use `--shape_family all` to run all families.
- `random`: sample validated shapes using the operator configuration.

## Correctness and performance

### Correctness

Generated kernels are checked against the PyTorch reference for output structure, shape, dtype, device placement, and numerical agreement. Numerical checks use operator-specific policies and precision-dependent tolerances. A task is considered correct only when the same generated implementation passes every required shape–precision configuration.

### Performance

Execution latency is measured after warm-up using backend-specific timing and device synchronization. Speedup is the reference latency divided by the generated kernel latency under the same configuration on the same backend; values above 1 indicate faster execution.

## Adding an operator

Create `operators/<category>/<operator>/` with the five required files shown in the repository layout. The reference module must provide:

- `class Model`
- `generate_inputs(shape)`

`helpers.py` must provide shape validation, and `shapes.py` must expose the shape definitions consumed by the benchmark runner. The registry discovers valid operator packages automatically.

To confirm discovery:

```bash
python -c "from operators.registry import list_operator_names; print(len(list_operator_names()))"
```

## Adding a backend

Create `backends/<name>_backend.py` and register the backend class with `@register_backend("<name>")` from `backends.backend_registry`. The runner imports this module when `--language <name>` is selected.

Use an existing backend as a reference and implement:

- Device selection via `setup_device(device_id)` or `set_device(device_id)` and `get_device()`.
- `compile(generated_code, op, dataset_name=None)`, returning `(success, error_message)` and exposing the generated `ModelNew` through `self.context`.
- `synchronize()`, `time_execution()`, and `cleanup()` for device synchronization, profiling, and resource cleanup.

For code generation, add the backend's hardware and SDK information in `codegen/runtime_env.py` and its implementation requirements in `codegen/backend_prompt_profiles.json`. Adapt prompt templates if the backend needs a different source layout or compilation contract.

## License

KernelVerse is released under the [MIT License](LICENSE). Third-party components retain their original licenses and copyright notices.
