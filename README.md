# NPUKernelBench

Ascend NPU kernel evaluation and benchmarking framework.

Evaluates custom NPU kernel solutions (PyTorch/torch_npu, Ascend C/CANN,
Triton) for correctness and performance by comparing them against
reference implementations.

## Installation

```bash
git clone <repo-url>
cd npu-kernelbench
uv sync
```

Requires Python >=3.10 and Ascend NPU hardware with torch (CPU version) and torch-npu installed from Huawei's repository. The two packages must use the same version.

## Quick Start

```bash
# Evaluate a single kernel (problem_dir mode)
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution tests/samples/matmul_triton/solution.json

# Evaluate with explicit paths
uv run npu-kernelbench eval \
    --definition tests/samples/gelu/definition.json \
    --workload tests/samples/gelu/workload.jsonl \
    --solution path/to/solution.json

# Run a full dataset level
uv run npu-kernelbench run-dataset data/kernel_generator --level level1

# Run with aggregated HTML report
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --html report.html

# Run multiple levels with a limit
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 --level level2 --limit 10

# Use isolated runner instead of persistent
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution solution.json --runner isolated

# Save traces to a specific file
uv run npu-kernelbench eval tests/samples/gelu \
    --solution solution.json -o my_traces.jsonl

# Save logs to file
uv run npu-kernelbench eval tests/samples/gelu \
    --solution solution.json --log-file eval.log --verbose
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `eval` | Evaluate a single kernel solution |
| `run-dataset` | Batch evaluate all kernels in a dataset directory |

### eval options

```
--definition PATH       Path to definition.json
--workload PATH         Path to workload.jsonl
--solution PATH         Path to solution.json (required)
--config PATH           Path to benchmark config JSON
--compile-timeout N     Compilation timeout in seconds (default: 120)
--timeout N             Evaluation subprocess timeout in seconds (default: 600)
--limit N               Max workloads to evaluate
--runner MODE           "persistent" (default) or "isolated"
--npu-ids IDS           Comma-separated NPU device IDs (default: 0)
--profile-dir PATH      Save profiling traces to this directory
--work-dir PATH         Working directory for staging (default: /tmp/npu-kernelbench)
--log-file PATH         Write logs to this file
--json                  Print trace JSON to stdout
--verbose, -v           Enable debug logging
-o, --output PATH       Write trace JSONL to this file (default: /tmp/npu-kernelbench/traces.jsonl)
--html PATH             Generate HTML report at the given path
--resume                Skip workloads/kernels already present in -o (resume after interruption)
```

### run-dataset options

```
-l, --level TEXT        Levels to evaluate (default: level1, repeatable)
--limit N               Max workloads per kernel
--solution-dir PATH     Directory containing solution files per kernel
--timeout N             Evaluation subprocess timeout in seconds (default: 600)
--runner MODE           "persistent" (default) or "isolated"
--npu-ids IDS           Comma-separated NPU device IDs (default: 0)
--profile-dir PATH      Save profiling traces to this directory
--work-dir PATH         Working directory for staging (default: /tmp/npu-kernelbench)
--log-file PATH         Write logs to this file
--verbose, -v           Show verbose output
-o, --output PATH       Write trace JSONL to this file (default: /tmp/npu-kernelbench/traces.jsonl)
--html PATH             Generate aggregated HTML report at the given path
--resume                Skip workloads/kernels already present in -o (resume after interruption)
```

Traces are always written to JSONL. Default path: `/tmp/npu-kernelbench/traces.jsonl`.
Use `-o` to specify a different path.

### Generate HTML Report

```bash
# Generate HTML report alongside the default JSONL output
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution solution.json --html report.html
```
Open `report.html` in any browser to view the results.

### Resume an Interrupted Run

```bash
# A batch run was interrupted — resume without redoing completed work.
# Workloads (and fully-done kernels) already present in -o are skipped;
# the output file keeps its existing results and appends only the new ones.
uv run npu-kernelbench run-dataset data/kernel_generator \
    --level level1 -o runs.jsonl --resume

# Works the same way for eval, and for incremental runs where new kernels
# were added to the dataset — only the missing workloads run.
uv run npu-kernelbench eval tests/samples/matmul_triton \
    --solution solution.json -o runs.jsonl --resume
```

Resume reads from the `-o` file (default `/tmp/npu-kernelbench/traces.jsonl`).
Reuse the same `-o` to continue a specific run; delete the file for a clean slate.

## Configuration

See `config.json` for all options:

```json
{
  "warmup_runs": 5,
  "iterations": 10,
  "seed": 200,
  "timeout": 600,
  "compile_timeout": 120,
  "benchmark_reference": true,
  "npu_id": 0,
  "npu_ids": [0],
  "runner": "persistent",
  "work_dir": "/tmp/npu-kernelbench",
  "profile_dir": "",
  "health_check_interval": 10,
  "max_solution_failures": 3
}
```

## Architecture

```
cli/main.py                  Click CLI (eval, run-dataset)
    │
    ▼
runner/                      Orchestration
├── runner.py                Runner ABC
├── isolated_runner.py       Process-per-evaluation (full NPU isolation)
├── persistent_runner.py     Long-lived worker processes (default)
└── eval.py                  Standalone eval driver (staged to work_dir)
    │
    ├── builder/             Transforms Solution → Runnable
    │   ├── python_builder.py   PyTorch, Triton, TileLang (importlib)
    │   └── ascendc_builder.py  Ascend C (CANN compilation)
    │
    ├── evaluator/           Evaluation pipeline
    │   ├── generator.py     Input tensor generation (gen_inputs, _rand_tensor)
    │   ├── correctness.py   Numerical correctness checking
    │   ├── performance.py   NPU profiler-based timing
    │   ├── anti_hack.py     Reward hack detection
    │   ├── evaluator.py     Evaluator ABC + BaselineHandle
    │   ├── default.py       DefaultEvaluator (general-purpose kernels)
    │   └── registry.py      EvaluatorRegistry (priority-ordered resolution)
    │
    ├── data/                Pydantic v2 models
    │   ├── definition.py    Definition (kernel spec, symbolic axes)
    │   ├── workload.py      Workload (InputSpec types, tolerances)
    │   ├── solution.py      Solution (source files, build spec)
    │   ├── trace.py         Trace, Evaluation, Correctness, Performance
    │   ├── shapes.py        Axis types (const, var, expr) + expression resolver
    │   └── dtypes.py        DType enum + torch/python dtype conversions
    │
    └── report/              Output formatting
        ├── formatter.py     Rich table output
        ├── score.py         Score computation
        ├── json_reporter.py JSON export
        └── html_reporter.py HTML report generation
```

### Execution flow

```
CLI → Runner pre_build (stage sources + eval files, compile)
    → Runner run_workload (per workload)
      → subprocess worker:
         → Builder loads Solution
         → Evaluator.build_baseline: gen_inputs → run reference
         → Evaluator.evaluate: run solution → correctness → performance → Trace
    → Report formats results (Rich table + score summary)
    → Traces saved to JSONL (default: work_dir/traces.jsonl)
```

### Standalone reproduction

After evaluation, `work_dir` contains all files needed to reproduce:

```
work_dir/
├── definition.json
├── workload.jsonl
├── solution.json
├── eval.py              ← cd work_dir && python eval.py
├── sources/
├── kernel.so            (AscendC only)
└── profile/
```

## Precision Standard

Operator evaluation and data validation in this repository uniformly adopt the
[CANN opbase experimental precision standard
(experimental_standard)](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md).

## Data Format

### definition.json

Defines a kernel's interface: symbolic axes, tensor specs, and reference implementation.

```json
{
  "name": "matmul_triton",
  "op_type": "matmul",
  "axes": {
    "M": {"type": "var", "description": "rows of A and C"},
    "K": {"type": "var", "description": "inner dimension"},
    "N": {"type": "var", "description": "cols of B and C"}
  },
  "inputs": {
    "a": {"shape": ["M", "K"], "dtype": "float32"},
    "b": {"shape": ["K", "N"], "dtype": "float32"}
  },
  "outputs": {
    "c": {"shape": ["M", "N"], "dtype": "float32"}
  },
  "reference": "import torch\ndef run(a, b):\n    return torch.mm(a, b)\n"
}
```

Axis types:
- `"const"` — fixed value, e.g. `{"type": "const", "value": 2048}`
- `"var"` — determined by workload at runtime
- `"expr"` — computed from other axes, e.g. `{"type": "expr", "expression": "M * 2"}`

### workload.jsonl (one JSON object per line)

Specifies concrete axis values and input data generation strategy.

```jsonl
{"axes": {"M": 64, "K": 128, "N": 64}, "inputs": {"a": {"type": "random", "dtype": "float32"}, "b": {"type": "random", "dtype": "float32"}}, "uuid": "mm-001"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "float16"}, "b": {"type": "random", "dtype": "float16"}}, "uuid": "mm-002"}
{"axes": {"M": 128, "K": 256, "N": 128}, "inputs": {"a": {"type": "random", "dtype": "bfloat16"}, "b": {"type": "random", "dtype": "bfloat16"}}, "uuid": "mm-003"}
```

InputSpec types:
- `{"type": "random", "dtype": "float16"}` — random tensor with optional dtype override
- `{"type": "scalar", "value": 2.5}` — scalar literal
- `{"type": "custom"}` — generated via `custom_inputs_entrypoint`
- `{"type": "safetensors", "path": "...", "tensor_key": "..."}` — loaded from safetensors file

### solution.json

A concrete kernel implementation.

```json
{
  "name": "my_matmul",
  "definition": "matmul_triton",
  "author": "developer",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["ascend_910b"],
    "entry_point": "kernel.py::matmul",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\nimport triton\n...\ndef matmul(a, b):\n    ...\n"
    }
  ]
}
```

Supported languages: `python`, `pytorch`, `triton`, `ascendc`.
Supported hardware: `ascend_910b`, `ascend_910c`, `LOCAL`.

## Sample Problems

The `tests/samples/` directory contains reference problems:

| Directory | Description | Language |
|-----------|-------------|----------|
| `gelu/` | GELU activation function | — |
| `identity/` | Identity function | — |
| `matmul_pytorch/` | Matrix multiplication | PyTorch |
| `matmul_triton/` | Matrix multiplication (Triton kernel) | Triton |
| `vector_add_ascendc/` | Vector addition | Ascend C |

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/npu_kernelbench/data/test_definition.py -v

# Run tests matching a keyword
uv run pytest tests/ -k "test_correctness" -v

# Run integration tests (requires NPU)
uv run pytest tests/test_integration.py -v
```

Tests marked with `requires_npu` or `ascend_c` will skip when those
dependencies aren't available.

## Build Wheel

```bash
pip install build
python -m build --wheel
```

## Kernel support status on hardware

| id | Level | kernel | A2/A3 | A5 |
|------|-------|----------|-------|----|
| 1 | level1 | Abs | ✅ | ✅ |
| 2 | level1 | AdamW | ✅ | ✅ |
| 3 | level1 | AdaptiveAvgPool3d | ✅ | ✅ |
| 4 | level1 | Add | ✅ | ✅ |
| 5 | level1 | AvgPool3d | ✅ | ✅ |
| 6 | level1 | Cat | ✅ | ✅ |
| 7 | level1 | CTCLoss | ✅ | ✅ |
| 8 | level1 | Cumsum | ✅ | ✅ |
| 9 | level1 | DynamicQuant | ✅ | ✅ |
| 10 | level1 | EmbeddingDenseBackward | ✅ | ✅ |
| 11 | level1 | Gather | ✅ | ✅ |
| 12 | level1 | GELU | ✅ | ✅ |
| 13 | level1 | GroupNorm | ✅ | ✅ |
| 14 | level1 | Histc | ✅ | ✅ |
| 15 | level1 | Index | ✅ | ✅ |
| 16 | level1 | IndexPut | ✅ | ✅ |
| 17 | level1 | Interpolate | ✅ | ✅ |
| 18 | level1 | IOU | ✅ | ✅ |
| 19 | level1 | LayerNorm | ✅ | ✅ |
| 20 | level1 | MaxPool3d | ✅ | ✅ |
| 21 | level1 | NLLLoss | ✅ | ✅ |
| 22 | level1 | NMS | ✅ | ❌ |
| 23 | level1 | Nonzero | ✅ | ✅ |
| 24 | level1 | Pad | ✅ | ✅ |
| 25 | level1 | Permute | ✅ | ✅ |
| 26 | level1 | Repeat | ✅ | ✅ |
| 27 | level1 | RepeatInterleave | ✅ | ✅ |
| 28 | level1 | Scatter | ✅ | ✅ |
| 29 | level1 | Sort | ✅ | ✅ |
| 30 | level1 | Split | ✅ | ✅ |
| 31 | level1 | Sum | ✅ | ✅ |
| 32 | level1 | SwiGLU | ✅ | ✅ |
| 33 | level1 | TopK | ✅ | ✅ |
| 34 | level2 | AdaptiveInstanceNormalization2DBackward | ✅ | ✅ |
| 35 | level2 | AdvanceStepFlashattn | ✅ | ❌ |
| 36 | level2 | AttentionSoftmaxWithSoftcappingAndDropout | ✅ | ✅ |
| 37 | level2 | Batched2DRopePositionEncodingBackward | ✅ | ✅ |
| 38 | level2 | DequantSwigluQuant | ✅ | ✅ |
| 39 | level2 | EmbeddingWithInitialLayernormBackward | ✅ | ✅ |
| 40 | level2 | FusedAddRmsnorm | ✅ | ✅ |
| 41 | level2 | FusedResidualRmsNormBackward | ✅ | ✅ |
| 42 | level2 | FusedRopeWithQkNormAndKvCacheUpdate | ✅ | ✅ |
| 43 | level2 | GaussianTopkSparseActivation | ✅ | ✅ |
| 44 | level2 | GroupNormSwish | ✅ | ✅ |
| 45 | level2 | HybridAttentionMaskPreparation | ✅ | ✅ |
| 46 | level2 | HyenaFftSizePaddingRfft | ✅ | ✅ |
| 47 | level2 | InterleaveRope | ✅ | ✅ |
| 48 | level2 | KvCacheUpdateWithRopeBackward | ✅ | ✅ |
| 49 | level2 | KvRmsnormRopeCache | ✅ | ✅ |
| 50 | level2 | MaskedSoftmaxWithAttentionDropoutBackward | ✅ | ✅ |
| 51 | level2 | MoeComputeExpertTokens | ✅ | ✅ |
| 52 | level2 | MoeFinalizeRouting | ✅ | ✅ |
| 53 | level2 | MoeGatingTopKSoftmax | ✅ | ✅ |
| 54 | level2 | MoeGroupScoreAggregationAndMasking | ✅ | ✅ |
| 55 | level2 | MoeInitRouting | ✅ | ✅ |
| 56 | level2 | MultiMaskAttentionAggregation | ✅ | ✅ |
| 57 | level2 | MultimodalRopePositionComputationWithGridBasedIndexing | ✅ | ✅ |
| 58 | level2 | QuantScatter | ✅ | ✅ |
| 59 | level2 | RotaryMul | ✅ | ✅ |
| 60 | level2 | SwigluQuant | ✅ | ✅ |
| 61 | level2 | TanhGatedResidualAddBackward | ✅ | ✅ |
| 62 | level2 | TimeDecayExponentialStabilization | ✅ | ✅ |
| 63 | level2 | TopKTopP | ✅ | ✅ |
| 64 | level3 | BatchMatmul | ✅ | ✅ |
| 65 | level3 | ConvDepthwise2d | ✅ | ✅ |
| 66 | level3 | ConvStandard1d | ✅ | ✅ |
| 67 | level3 | ConvStandard2d | ✅ | ✅ |
| 68 | level3 | ConvStandard3d | ✅ | ✅ |
| 69 | level3 | ConvTranspose2d | ✅ | ✅ |
| 70 | level3 | GroupedMatmul | ✅ | ✅ |
| 71 | level3 | MatmulBothTrans | ✅ | ✅ |
| 72 | level3 | MatmulTransA | ✅ | ✅ |
| 73 | level3 | MatmulTransB | ✅ | ✅ |
| 74 | level4 | FFN | ✅ | ✅ |
| 75 | level4 | FluxTimestepGuidanceProjectionEmbedding | ✅ | ✅ |
| 76 | level4 | FusedInferAttentionScore | ✅ | ✅ |
| 77 | level4 | FusionAttention | ✅ | ✅ |
| 78 | level4 | GroupedMatmulSwigluQuant | ✅ | ✅ |
| 79 | level4 | GroupedMatmulSwigluQuantV2 | ✅ | ❌ |
| 80 | level4 | LightningIndexer | ✅ | ✅ |
| 81 | level4 | LSTM | ✅ | ❌ |
| 82 | level4 | QuantMatmul | ✅ | ✅ |
| 83 | level4 | QuantMatmulReduceSum | ✅ | ✅ |
| 84 | level4 | SparseFlashAttention | ✅ | ✅ |
| 85 | level4 | WeightQuantBatchmatmul | ✅ | ✅ |

## Acknowledgements

NPUKernelBench's overall architecture and several implementation patterns
(notably input tensor generation) draw inspiration from
[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench)
([docs](https://bench.flashinfer.ai/docs/)), an open-source benchmark suite for
LLM kernels released under the Apache 2.0 License.
