#!/usr/bin/env python3

"""NPU KernelBench evaluation driver.

Self-contained script written to the NPU staging directory.
Evaluates a user solution against workloads and outputs JSONL Trace
objects to stdout. All non-JSON output goes to stderr.

Usage:
    cd /tmp/npu-kernelbench/{problem}-{solution}-{author}
    python eval.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ── stdout redirect: reserve original stdout for JSON output ───────────────
_real_stdout_fd = os.dup(1)
os.dup2(2, 1)  # fd 1 → stderr
sys.stdout = open(1, "w", buffering=1, closefd=False)
_real_stdout = os.fdopen(_real_stdout_fd, "w", buffering=1)

import torch  # noqa: E402 — must come after redirect

# ── Staging directory ──────────────────────────────────────────────────────
STAGING_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(STAGING_DIR))


def _emit(trace_dict: dict) -> None:
    """Write a single JSONL line to the real stdout."""
    _real_stdout.write(json.dumps(trace_dict) + "\n")
    _real_stdout.flush()


def _get_hardware_name() -> str:
    try:
        if torch.npu.is_available():
            name = torch.npu.get_device_name()
            if name:
                return name
    except Exception:
        pass
    return os.environ.get("NPU_HARDWARE_NAME", "Ascend_Unknown")


# ── Load problem files ─────────────────────────────────────────────────────
def _load_problem() -> tuple[dict, list[dict], dict | None]:
    local_def = STAGING_DIR / "definition.json"
    local_wkl = STAGING_DIR / "workload.jsonl"
    local_cfg = STAGING_DIR / "config.json"

    if not local_def.exists():
        raise RuntimeError("definition.json not found in staging directory")
    if not local_wkl.exists():
        raise RuntimeError("workload.jsonl not found in staging directory")

    definition = json.loads(local_def.read_text())
    workloads = []
    for line in local_wkl.read_text().splitlines():
        line = line.strip()
        if line:
            workloads.append(json.loads(line))

    config = None
    if local_cfg.exists():
        config = json.loads(local_cfg.read_text())

    return definition, workloads, config


definition_dict, workload_dicts, config_dict = _load_problem()

# ── Imports from npu_kernelbench ───────────────────────────────────────────
from npu_kernelbench.cli.config import BenchmarkConfig  # noqa: E402
from npu_kernelbench.data.definition import Definition  # noqa: E402
from npu_kernelbench.data.solution import Solution  # noqa: E402
from npu_kernelbench.data.trace import (  # noqa: E402
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from npu_kernelbench.data.workload import Workload  # noqa: E402
from npu_kernelbench.evaluator.correctness import (  # noqa: E402
    check_tensor_sanity,
    compute_error_stats,
    set_seed,
)
from npu_kernelbench.evaluator.generator import gen_inputs  # noqa: E402
from npu_kernelbench.evaluator.performance import profile_kernel  # noqa: E402
from npu_kernelbench.evaluator.anti_hack import (  # noqa: E402
    run_ast_scan,
    detect_kernel_signature_mismatch,
    detect_output_caching,
    detect_monkey_patch,
    detect_clock_anomaly,
    RewardHackResult,
)

# ── Parse models ───────────────────────────────────────────────────────────
definition = Definition(**definition_dict)
workloads = [Workload(**w) for w in workload_dicts]
config = BenchmarkConfig(**config_dict) if config_dict else BenchmarkConfig()

# ── Parse solution ─────────────────────────────────────────────────────────
solution = Solution(**json.loads((STAGING_DIR / "solution.json").read_text()))

# ── Exec reference code ────────────────────────────────────────────────────
_ref_file = STAGING_DIR / "_reference.py"
_ref_file.write_text(definition.reference)
try:
    ref_spec = importlib.util.spec_from_file_location("_reference", _ref_file)
    ref_module = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref_module)
except Exception as e:
    raise RuntimeError(f"Failed to exec reference code: {e}") from e

ref_fn = ref_module.__dict__.get("run")
if ref_fn is None:
    raise RuntimeError("Reference code does not define a top-level 'run' function")

# ── Load user solution ─────────────────────────────────────────────────────
entry_file = solution.get_entry_path()
entry_symbol = solution.get_entry_symbol()
sources_dir = STAGING_DIR / "sources" if (STAGING_DIR / "sources").exists() else STAGING_DIR
module_path = sources_dir / entry_file

sol_spec = importlib.util.spec_from_file_location(
    "npu_kernelbench_eval_user_solution", str(module_path)
)
if sol_spec is None:
    raise RuntimeError(f"Could not load solution module from {module_path}")

sol_module = importlib.util.module_from_spec(sol_spec)
sys.modules["npu_kernelbench_eval_user_solution"] = sol_module
sol_spec.loader.exec_module(sol_module)
user_fn = getattr(sol_module, entry_symbol, None)
if user_fn is None:
    raise RuntimeError(
        f"Entry point '{entry_symbol}' not found in {entry_file}"
    )

# ── Load .so if present (AscendC) ──────────────────────────────────────────
for so_file in STAGING_DIR.glob("*.so"):
    torch.ops.load_library(str(so_file))

# ── Main eval loop ─────────────────────────────────────────────────────────
device = f"npu:{config.npu_id}"
hw_name = _get_hardware_name()
exit_code = 0

# L0: Anti-hack — AST static scan (once before workload loop)
l1_hacks: list[str] = []
if "ast_scan" in config.enable_anti_hack:
    sources_dir = STAGING_DIR / "sources"
    if not sources_dir.exists():
        sources_dir = STAGING_DIR
    py_files = list(sources_dir.glob("*.py"))
    if py_files:
        l1_hacks = run_ast_scan(py_files)


def _build_log(hack_lines: list[str], main_msg: str) -> str:
    """Join hack descriptions with the main log message."""
    if hack_lines:
        return "\n".join(hack_lines) + "\n" + main_msg
    return main_msg


for workload in workloads:
    set_seed(config.seed)
    hacks = list(l1_hacks)  # per-workload copy

    # 1. Generate inputs
    try:
        inputs = gen_inputs(definition, workload, device, config.seed)
    except Exception as e:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.RUNTIME_ERROR,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=_build_log(hacks, f"gen_inputs failed: {e}\n{traceback.format_exc()}"),
            ),
        )
        _emit(trace.model_dump())
        exit_code = 1
        continue

    # 2. Run reference
    try:
        ref_result = ref_fn(*inputs)
        if not isinstance(ref_result, (tuple, list)):
            ref_result = [ref_result]
        ref_outputs = dict(zip(definition.outputs.keys(), ref_result))
    except Exception as e:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.RUNTIME_ERROR,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=_build_log(hacks, f"Reference run() failed: {e}\n{traceback.format_exc()}"),
            ),
        )
        _emit(trace.model_dump())
        exit_code = 1
        continue

    # 3. Run user solution
    try:
        sol_result = user_fn(*inputs)
        if not isinstance(sol_result, (tuple, list)):
            sol_result = [sol_result]
        sol_outputs = dict(zip(definition.outputs.keys(), sol_result))
    except Exception as e:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.RUNTIME_ERROR,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=_build_log(hacks, f"User function failed: {e}\n{traceback.format_exc()}"),
            ),
        )
        _emit(trace.model_dump())
        exit_code = 1
        continue

    # 4. Shape check
    shape_error = None
    for name in definition.outputs.keys():
        if sol_outputs[name].shape != ref_outputs[name].shape:
            shape_error = (
                f"Output '{name}': sol shape {sol_outputs[name].shape} "
                f"!= ref shape {ref_outputs[name].shape}"
            )
            break
    if shape_error:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.INCORRECT_SHAPE,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=_build_log(hacks, shape_error),
            ),
        )
        _emit(trace.model_dump())
        exit_code = 1
        continue

    # 5. Dtype check
    dtype_error = None
    for name in definition.outputs.keys():
        if sol_outputs[name].dtype != ref_outputs[name].dtype:
            dtype_error = (
                f"Output '{name}': sol dtype {sol_outputs[name].dtype} "
                f"!= ref dtype {ref_outputs[name].dtype}"
            )
            break
    if dtype_error:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.INCORRECT_DTYPE,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=_build_log(hacks, dtype_error),
            ),
        )
        _emit(trace.model_dump())
        exit_code = 1
        continue

    # 6. Numerical correctness check
    tol = workload.tolerance
    correctness = Correctness()
    numerically_wrong = False
    for i, name in enumerate(definition.outputs.keys()):
        sol = sol_outputs[name]
        ref = list(ref_outputs.values())[i]

        sanity = check_tensor_sanity(
            sol, ref, allow_negative_inf=tol.allow_negative_inf
        )
        if sanity is not None:
            # Sanity failure (NaN/Inf). Record the sanity result and fall
            # through to the single emission below — emitting here would
            # double up with the post-loop ``if numerically_wrong`` block.
            correctness = sanity
            numerically_wrong = True
            exit_code = 1
            break

        c, exceeds = compute_error_stats(sol, ref, tol)
        if c.max_absolute_error > correctness.max_absolute_error:
            correctness = c
        if c.has_nan:
            correctness = c
        elif c.has_inf and not correctness.has_nan:
            correctness = c
        if exceeds:
            numerically_wrong = True
            exit_code = 1
            break

    if numerically_wrong:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                correctness=correctness,
                log=_build_log(hacks, ""),
            ),
        )
        _emit(trace.model_dump())
        continue

    # 7. Performance timing
    try:
        timing = profile_kernel(
            user_fn,
            tuple(inputs),
            npu_id=config.npu_id,
            warmup_runs=config.warmup_runs,
            iterations=config.iterations,
            profile_dir=str(STAGING_DIR / "profile" / f"{workload.uuid}_sol"),
        )
        sol_latency = timing.mean_ms

        # L2: kernel signature check
        if "kernel_signature" in config.enable_anti_hack:
            profile_dir = str(STAGING_DIR / "profile" / f"{workload.uuid}_sol")
            hacks.extend(
                detect_kernel_signature_mismatch(sources_dir, profile_dir)
            )

        # L3: reward-hack checks
        if "output_caching" in config.enable_anti_hack:
            try:
                def _input_generator(seed: int) -> tuple:
                    return tuple(
                        gen_inputs(definition, workload, device, seed)
                    )

                result = detect_output_caching(user_fn, _input_generator)
                if result.detected:
                    hacks.append(
                        f"Reward hack ({result.hack_type}): {result.evidence}"
                    )
            except Exception:
                pass

        if "monkey_patch" in config.enable_anti_hack and definition.reference_hash:
            result = detect_monkey_patch(
                definition.reference, definition.reference_hash
            )
            if result.detected:
                hacks.append(
                    f"Reward hack ({result.hack_type}): {result.evidence}"
                )

        if "clock_anomaly" in config.enable_anti_hack and timing.durations:
            result = detect_clock_anomaly(timing.durations)
            if result.detected:
                hacks.append(
                    f"Reward hack ({result.hack_type}): {result.evidence}"
                )
    except Exception as e:
        trace = Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=Evaluation(
                status=EvaluationStatus.RUNTIME_ERROR,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                log=_build_log(hacks, f"Timing failed: {e}\n{traceback.format_exc()}"),
            ),
        )
        _emit(trace.model_dump())
        exit_code = 1
        continue

    # 8. Reference latency (optional)
    ref_latency = 0.0
    if config.benchmark_reference:
        try:
            ref_timing = profile_kernel(
                ref_fn,
                tuple(inputs),
                npu_id=config.npu_id,
                warmup_runs=config.warmup_runs,
                iterations=config.iterations,
                profile_dir=str(STAGING_DIR / "profile" / f"{workload.uuid}_ref"),
            )
            ref_latency = ref_timing.mean_ms
        except Exception:
            pass

    speedup = ref_latency / max(sol_latency, 1e-9)

    # 9. Emit final trace
    if hacks:
        status = EvaluationStatus.REWARD_HACK
    else:
        status = EvaluationStatus.PASSED

    trace = Trace(
        definition=definition.name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=status,
            environment=Environment(hardware=hw_name),
            timestamp=datetime.now(timezone.utc).isoformat(),
            correctness=correctness,
            performance=Performance(
                latency_ms=sol_latency,
                reference_latency_ms=ref_latency,
                speedup_factor=speedup,
                durations=timing.durations,
            ),
            log="\n".join(hacks) if hacks else "",
        ),
    )
    _emit(trace.model_dump())

sys.exit(exit_code)
