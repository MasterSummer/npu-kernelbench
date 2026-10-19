"""DefaultEvaluator — general-purpose numerical correctness checker."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import CustomInput, SafetensorsInput, Workload
from npu_kernelbench.data.trace import Correctness
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.evaluator.correctness import (
    set_seed,
    as_tensor_sequence,
    check_tensor_sanity,
    compute_error_stats,
)
from npu_kernelbench.evaluator.evaluator import Evaluator, BaselineHandle
from npu_kernelbench.evaluator.generator import gen_inputs, load_safetensors


class DefaultEvaluator(Evaluator):
    """Handles general-purpose kernels: GEMM, RMSNorm, RoPE, etc.

    Falls back to per-element atol + rtol * |ref| with matched_ratio check.
    Always returns True from can_evaluate() — this is the catch-all evaluator.
    """

    def can_evaluate(self, definition: Definition) -> bool:
        return True

    def build_baseline(
        self,
        definition: Definition,
        workload: Workload,
        device: str,
        config: BenchmarkConfig,
    ) -> BaselineHandle:
        """Generate inputs, run reference, return baseline."""
        set_seed(config.seed)

        device_obj = torch.device(device)

        # Determine if safetensors or custom inputs are needed
        has_safetensors = any(
            isinstance(s, SafetensorsInput) for s in workload.inputs.values()
        )
        has_custom = any(
            isinstance(s, CustomInput) for s in workload.inputs.values()
        )

        # Load safetensors
        safe_tensors = (
            load_safetensors(definition, workload) if has_safetensors else {}
        )

        # Prepare custom inputs function
        custom_fn = None
        if has_custom:
            custom_fn = _load_custom_inputs_fn(definition, device_obj)

        # Generate inputs as ordered list
        inputs = gen_inputs(
            definition, workload, device, config.seed,
            safe_tensors=safe_tensors, custom_inputs_fn=custom_fn,
        )

        # Run reference — stores outputs keyed by definition output names
        ref_outputs = _run_reference(
            definition.reference, inputs, list(definition.outputs.keys())
        )

        # Optionally time reference
        ref_latency = None
        if config.benchmark_reference:
            from npu_kernelbench.evaluator.performance import profile_kernel

            def _ref_fn(*args):
                return _run_reference(definition.reference, list(args))

            timing = profile_kernel(
                _ref_fn,
                tuple(inputs),
                npu_id=config.npu_id,
                warmup_runs=config.warmup_runs,
                iterations=config.iterations,
                profile_dir=config.profile_dir or f"{config.work_dir}/profile",
            )
            ref_latency = timing.mean_ms

        return BaselineHandle(
            definition_name=definition.name,
            device=device,
            workload_uuid=workload.uuid,
            inputs=inputs,
            outputs=ref_outputs,
            mean_latency_ms=ref_latency,
        )

    def check_correctness(
        self,
        definition: Definition,
        workload: Workload,
        baseline: BaselineHandle,
        solution_outputs: dict[str, torch.Tensor],
    ) -> tuple[Correctness, bool]:
        """Per-element numerical error check using tolerance from workload.

        Iterates over output tensors by definition output order. Compares
        each solution output against the corresponding baseline output.
        """
        tol = workload.tolerance
        max_abs = 0.0
        max_rel = 0.0

        for i, name in enumerate(definition.outputs.keys()):
            sol = solution_outputs[name]
            ref = list(baseline.outputs.values())[i]

            # Variadic outputs (e.g. nonzero as_tuple=True) are packed as a
            # tuple under one output name; compare element-wise.
            sol_seq = as_tensor_sequence(sol)
            ref_seq = as_tensor_sequence(ref)
            for s, r in zip(sol_seq, ref_seq):
                sanity = check_tensor_sanity(
                    s, r, allow_negative_inf=tol.allow_negative_inf
                )
                if sanity is not None:
                    return sanity, False

                correctness, exceeds = compute_error_stats(s, r, tol)
                max_abs = max(max_abs, correctness.max_absolute_error)
                max_rel = max(max_rel, correctness.max_relative_error)
                if exceeds:
                    return correctness, False

        return Correctness(
            max_absolute_error=max_abs, max_relative_error=max_rel
        ), True


def _run_reference(
    reference_code: str,
    inputs: list,
    output_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Execute reference implementation and return output tensors.

    When output_names is provided, returns a dict keyed by output name.
    Otherwise falls back to positional index strings ('0', '1', ...).
    """
    local_ns: dict = {}
    exec(reference_code, local_ns)
    run_fn = local_ns["run"]
    result = run_fn(*inputs)
    if not isinstance(result, (tuple, list)):
        result = [result]
    if output_names:
        if len(output_names) == len(result):
            # 1:1 mapping — each declared output name gets one tensor
            return dict(zip(output_names, result))
        if len(output_names) == 1:
            # Variadic output: the reference returned a different number of
            # tensors than the single declared output (e.g.
            # torch.nonzero(as_tuple=ue) on n-D input returns n tensors).
            # Pack the whole sequence under the single declared output name
            # and preserve it as a tuple so downstream shape/dtype/correctness
            # checks can compare element-wise.
            return {output_names[0]: tuple(result)}
    return {str(i): t for i, t in enumerate(result)}


def _load_custom_inputs_fn(
    definition: Definition,
    device: torch.device,
):
    """Extract custom_inputs_entrypoint from definition.reference and return a callable.

    Returns a function with signature
    (resolved_axes: dict, device: torch.device) -> dict[str, Tensor].
    """
    entrypoint = definition.custom_inputs_entrypoint
    if not entrypoint:
        raise RuntimeError(
            f"Definition '{definition.name}' has CustomInput but "
            f"no custom_inputs_entrypoint"
        )

    local_ns: dict = {}
    exec(definition.reference, local_ns)

    if entrypoint not in local_ns:
        raise RuntimeError(
            f"custom_inputs_entrypoint '{entrypoint}' not found in "
            f"reference code for definition '{definition.name}'"
        )

    custom_fn = local_ns[entrypoint]

    def wrapper(resolved_axes: dict, dev: torch.device) -> dict:
        return custom_fn(resolved_axes, dev)

    return wrapper
