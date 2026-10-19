"""Evaluator ABC — base class for kernel-type-specific evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.data.trace import (
    Correctness, Performance, Evaluation, EvaluationStatus,
    Environment, Trace,
)
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.evaluator.correctness import as_tensor_sequence


@dataclass
class BaselineHandle:
    """Pre-computed reference data shared across evaluations.

    Built once per (definition, workload) pair and reused across all
    solutions evaluating against the same workload.
    """
    definition_name: str
    device: str
    workload_uuid: str
    inputs: list = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    mean_latency_ms: Optional[float] = None


class Evaluator(ABC):
    """Abstract evaluator for a specific kernel type.

    Subclasses implement kernel-type-specific correctness checking.
    The template method evaluate() orchestrates the full pipeline.
    """

    @abstractmethod
    def can_evaluate(self, definition: Definition) -> bool:
        """Return True if this evaluator handles the given definition."""
        ...

    @abstractmethod
    def build_baseline(
        self,
        definition: Definition,
        workload: Workload,
        device: str,
        config: BenchmarkConfig,
    ) -> BaselineHandle:
        """Generate inputs, run reference implementation, capture outputs.

        The baseline is cached by the caller (Runner) and reused across
        multiple solution evaluations against the same workload.
        """
        ...

    @abstractmethod
    def check_correctness(
        self,
        definition: Definition,
        workload: Workload,
        baseline: BaselineHandle,
        solution_outputs: dict[str, torch.Tensor],
    ) -> tuple[Correctness, bool]:
        """Check solution outputs against baseline.

        Returns (correctness_metrics, is_pass).
        is_pass=True means outputs are correct within tolerance.
        """
        ...

    def eval_performance(
        self,
        solution_fn: Callable,
        inputs: list,
        definition: Definition,
        config: BenchmarkConfig,
    ) -> Performance:
        """Measure solution execution time.

        Default: use timing.profile_kernel(). Override for custom timing
        (e.g., sampling needs multiple trials for statistical validity).
        """
        from npu_kernelbench.evaluator.performance import profile_kernel

        input_args = tuple(inputs)
        timing = profile_kernel(
            solution_fn,
            input_args,
            npu_id=config.npu_id,
            warmup_runs=config.warmup_runs,
            iterations=config.iterations,
            profile_dir=config.profile_dir or f"{config.work_dir}/profile",
        )
        return Performance(
            latency_ms=timing.mean_ms,
            durations=timing.durations,
        )

    def evaluate(
        self,
        definition: Definition,
        workload: Workload,
        solution_fn: Callable,
        baseline: BaselineHandle,
        config: BenchmarkConfig,
    ) -> Trace:
        """Template method: baseline -> correctness -> performance -> Trace.

        Orchestrates the full evaluation pipeline. Subclasses typically
        only override can_evaluate(), build_baseline(), check_correctness(),
        and optionally eval_performance().
        """
        from datetime import datetime, timezone
        import os
        from pathlib import Path
        from npu_kernelbench.evaluator.anti_hack import (
            run_ast_scan,
            detect_kernel_signature_mismatch,
            detect_output_caching,
            detect_monkey_patch,
            detect_clock_anomaly,
            RewardHackResult,
        )
        from npu_kernelbench.evaluator.generator import gen_inputs
        import logging

        logger = logging.getLogger(__name__)

        hw_name = _get_hardware_name()

        # L0: Anti-hack — AST static scan
        hacks: list[str] = []
        if "ast_scan" in config.enable_anti_hack and config.solution_source_dir:
            source_dir = Path(config.solution_source_dir)
            if source_dir.exists():
                py_files = list(source_dir.glob("*.py"))
                if py_files:
                    hacks.extend(run_ast_scan(py_files))

        # 1. Run solution
        try:
            input_args = tuple(baseline.inputs)
            sol_raw = solution_fn(*input_args)
            if not isinstance(sol_raw, (tuple, list)):
                sol_raw = [sol_raw]
            sol_outputs_list = list(sol_raw)
        except Exception as e:
            return Trace(
                definition=definition.name,
                workload=workload,
                solution=getattr(solution_fn, "__name__", "unknown"),
                evaluation=Evaluation(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    environment=Environment(hardware=hw_name),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    log=_join_log(hacks, str(e)),
                ),
            )

        # Normalize to dict — when the solution returns a variadic sequence
        # whose length difrs from the declared outputs (e.g. nonzero with
        # as_tuple=True on n-D input), pack the whole sequence under the
        # single declared output name, mirroring _run_reference's packing.
        out_names = list(definition.outputs.keys())
        if len(out_names) == len(sol_outputs_list):
            sol_outputs = dict(zip(out_names, sol_outputs_list))
        elif len(out_names) == 1:
            sol_outputs = {out_names[0]: tuple(sol_outputs_list)}
        else:
            sol_outputs = {str(i): t for i, t in enumerate(sol_outputs_list)}

        # 2. Shape check — variadic outputs (packed as tuple) are compared
        # element-wise; a length mismatch is itself a shape error.
        ref_outputs = baseline.outputs
        for name in definition.outputs:
            sol_seq = as_tensor_sequence(sol_outputs[name])
            ref_seq = as_tensor_sequence(ref_outputs[name])
            if len(sol_seq) != len(ref_seq):
                return Trace(
                    definition=definition.name,
                    workload=workload,
                    solution=getattr(solution_fn, "__name__", "unknown"),
                    evaluation=Evaluation(
                        status=EvaluationStatus.INCORRECT_SHAPE,
                        environment=Environment(hardware=hw_name),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        log=_join_log(hacks, (
                            f"Output '{name}': sol has {len(sol_seq)} element(s) "
                            f"!= ref has {len(ref_seq)} element(s)"
                        )),
                    ),
                )

        # 2b. Per-element shape check — each corresponding tensor must match
        # in shape; otherwise compute_error_stats would raise on ``x - y``.
        for name in definition.outputs:
            sol_seq = as_tensor_sequence(sol_outputs[name])
            ref_seq = as_tensor_sequence(ref_outputs[name])
            for j, (s, r) in enumerate(zip(sol_seq, ref_seq)):
                s_shape = tuple(getattr(s, "shape", ()))
                r_shape = tuple(getattr(r, "shape", ()))
                if s_shape != r_shape:
                    return Trace(
                        definition=definition.name,
                        workload=workload,
                        solution=getattr(solution_fn, "__name__", "unknown"),
                        evaluation=Evaluation(
                            status=EvaluationStatus.INCORRECT_SHAPE,
                            environment=Environment(hardware=hw_name),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            log=_join_log(hacks, (
                                f"Output '{name}[{j}]': sol shape {s_shape} "
                                f"!= ref shape {r_shape}"
                            )),
                        ),
                    )

        # 3. Dtype check — variadic outputs compared element-wise.
        for name in definition.outputs:
            sol_seq = as_tensor_sequence(sol_outputs[name])
            ref_seq = as_tensor_sequence(ref_outputs[name])
            for j, (s, r) in enumerate(zip(sol_seq, ref_seq)):
                if s.dtype != r.dtype:
                    return Trace(
                        definition=definition.name,
                        workload=workload,
                        solution=getattr(solution_fn, "__name__", "unknown"),
                        evaluation=Evaluation(
                            status=EvaluationStatus.INCORRECT_DTYPE,
                            environment=Environment(hardware=hw_name),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            log=_join_log(hacks, (
                                f"Output '{name}[{j}]': sol dtype {s.dtype} "
                                f"!= ref dtype {r.dtype}"
                            )),
                        ),
                    )

        # 4. Correctness check (delegates to subclass)
        correctness, is_pass = self.check_correctness(
            definition, workload, baseline, sol_outputs
        )

        if not is_pass:
            return Trace(
                definition=definition.name,
                workload=workload,
                solution=getattr(solution_fn, "__name__", "unknown"),
                evaluation=Evaluation(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    environment=Environment(hardware=hw_name),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    correctness=correctness,
                    log="\n".join(hacks) if hacks else "",
                ),
            )

        perf = self.eval_performance(
            solution_fn, baseline.inputs, definition, config
        )
        if baseline.mean_latency_ms:
            perf.reference_latency_ms = baseline.mean_latency_ms
            perf.speedup_factor = baseline.mean_latency_ms / max(perf.latency_ms, 1e-9)

        # L2/L3: Anti-hack — kernel signature + reward-hack checks
        if "kernel_signature" in config.enable_anti_hack:
            profile_dir = config.profile_dir or f"{config.work_dir}/profile"
            sig_hacks = detect_kernel_signature_mismatch(
                config.solution_source_dir, profile_dir
            )
            logger.warning(
                "[anti_hack] kernel_signature mismatch count=%d, hacks=%s",
                len(sig_hacks),
                sig_hacks,
            )
            hacks.extend(sig_hacks)

        if "output_caching" in config.enable_anti_hack:
            try:
                def _input_generator(seed: int) -> tuple:
                    return tuple(
                        gen_inputs(
                            definition, workload, baseline.device, seed
                        )
                    )

                result = detect_output_caching(solution_fn, _input_generator)
                logger.warning(
                    "[anti_hack] output_caching detected=%s evidence=%s",
                    result.detected,
                    result.evidence,
                )
                if result.detected:
                    hacks.append(
                        f"Reward hack ({result.hack_type}): "
                        f"{result.evidence}"
                    )
            except Exception as e:
                logger.warning("[anti_hack] output_caching check failed: %s", e)

        if "monkey_patch" in config.enable_anti_hack and definition.reference_hash:
            result = detect_monkey_patch(
                definition.reference, definition.reference_hash
            )
            logger.warning(
                "[anti_hack] monkey_patch detected=%s evidence=%s",
                result.detected,
                result.evidence,
            )
            if result.detected:
                hacks.append(
                    f"Reward hack ({result.hack_type}): "
                    f"{result.evidence}"
                )

        if "clock_anomaly" in config.enable_anti_hack and perf.durations:
            result = detect_clock_anomaly(perf.durations)
            logger.warning(
                "[anti_hack] clock_anomaly detected=%s evidence=%s",
                result.detected,
                result.evidence,
            )
            if result.detected:
                hacks.append(
                    f"Reward hack ({result.hack_type}): "
                    f"{result.evidence}"
                )

        # Determine final status
        if hacks:
            status = EvaluationStatus.REWARD_HACK
        else:
            status = EvaluationStatus.PASSED

        return Trace(
            definition=definition.name,
            workload=workload,
            solution=getattr(solution_fn, "__name__", "unknown"),
            evaluation=Evaluation(
                status=status,
                environment=Environment(hardware=hw_name),
                timestamp=datetime.now(timezone.utc).isoformat(),
                correctness=correctness,
                performance=perf,
                log="\n".join(hacks) if hacks else "",
            ),
        )


def _get_hardware_name() -> str:
    """Detect NPU hardware name at runtime."""
    import os
    try:
        if torch.npu.is_available():
            name = torch.npu.get_device_name()
            if name:
                return name
    except Exception:
        pass
    return os.environ.get("NPU_HARDWARE_NAME", "Ascend_Unknown")


def _join_log(hacks: list[str], msg: str) -> str:
    """Join hack descriptions with a main log message."""
    if hacks:
        return "\n".join(hacks) + "\n" + msg
    return msg
