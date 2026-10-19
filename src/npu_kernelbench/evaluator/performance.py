"""NPU kernel performance timing via torch_npu profiler."""

from __future__ import annotations

import csv
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch_npu  # noqa: F401 -- register NPU device


@dataclass
class TimingResult:
    """Timing statistics from NPU kernel profiling."""

    mean_ms: float
    durations: list[float] = field(default_factory=list)


def parse_kernel_durations(trace_dir: str, iterations: int = 1) -> float | None:
    """Extract per-iteration mean kernel duration (ms) from profiler trace.

    Primary source: step_trace_time.csv ``Computing`` column divided by
    iterations.  Fallback: kernel_details.csv total duration / iterations.
    Returns None if no trace data is available.
    """
    trace_path = Path(trace_dir)

    # Primary: step_trace_time.csv
    step_files = list(trace_path.glob("**/step_trace_time.csv"))
    if step_files:
        return _parse_trace_csv(step_files[0], iterations, ["Computing"])

    # Fallback: kernel_details.csv
    kernel_files = list(trace_path.glob("**/kernel_details.csv"))
    if kernel_files:
        return _parse_trace_csv(kernel_files[0], iterations, ["Duration(us)", "Duration"])

    return None


def _parse_trace_csv(
    csv_path: Path, iterations: int, columns: list[str],
) -> float | None:
    """Sum column values (µs) across all rows, return per-iteration mean (ms).

    Columns are tried in order for each row; the first non-empty value is used.
    """
    total_us = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            val = ""
            for col in columns:
                val = row.get(col, "")
                if val:
                    break
            if val:
                try:
                    total_us += float(val.strip())
                except (ValueError, TypeError):
                    pass
    if total_us == 0.0:
        return None
    return total_us / iterations / 1000.0


def _event_based_timing(
    kernel_fn: Callable,
    inputs: tuple,
    device: torch.device,
    iterations: int,
) -> list[float]:
    """Fallback: NPU Event wall-clock timing.

    Used when profiler trace parsing returns no data.
    """
    durations: list[float] = []
    for _ in range(iterations + 1):  # +1 for cold-start, discarded
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        kernel_fn(*inputs)
        end_event.record()
        torch.npu.synchronize(device)
        elapsed_ms = start_event.elapsed_time(end_event)
        durations.append(elapsed_ms)
    return durations[1:]  # discard cold-start first run


def profile_kernel(
    kernel_fn: Callable,
    inputs: tuple,
    *,
    npu_id: int = 0,
    warmup_runs: int = 10,
    iterations: int = 50,
    profile_dir: str | None = None,
) -> TimingResult:
    """Profile an NPU kernel using torch_npu profiler as primary timing source.

    Reads step_trace_time.csv Computing time divided by iterations.
    Falls back to kernel_details.csv, then NPU Event wall-clock timing.

    Args:
        kernel_fn: Callable that takes inputs and runs the kernel.
        inputs: Tuple of tensor arguments to pass to kernel_fn.
        npu_id: NPU device ID.
        warmup_runs: Number of warmup iterations.
        iterations: Number of timed iterations.
        profile_dir: If set, persist profiling traces to this directory.
                     Otherwise, use a temp directory (auto-cleaned).

    Returns:
        TimingResult with per-iteration NPU kernel durations.
    """
    device = torch.device(f"npu:{npu_id}")

    use_temp = profile_dir is None
    trace_dir = profile_dir if profile_dir else tempfile.mkdtemp(
        prefix="npu-profile-"
    )
    mean_ms: float | None = None
    durations: list[float] = []

    try:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )

        total_steps = warmup_runs + iterations
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup_runs, active=iterations,
                repeat=1, skip_first=0,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                trace_dir
            ),
            profile_memory=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(total_steps):
                kernel_fn(*inputs)
                torch.npu.synchronize(device)
                prof.step()

        mean_ms = parse_kernel_durations(trace_dir, iterations)

        # Build per-iteration durations from step_trace_time.csv when available.
        trace_path = Path(trace_dir)
        step_files = list(trace_path.glob("**/step_trace_time.csv"))
        if step_files:
            with open(step_files[0], newline="") as f:
                for row in csv.DictReader(f):
                    val = row.get("Computing", "")
                    if val:
                        try:
                            durations.append(float(val.strip()) / 1000.0)
                        except (ValueError, TypeError):
                            pass
    except Exception:
        pass
    finally:
        if use_temp:
            shutil.rmtree(trace_dir, ignore_errors=True)

    # Primary result from profiler trace
    if mean_ms is not None:
        return TimingResult(mean_ms=mean_ms, durations=durations or [mean_ms])

    # Fallback to NPU Event wall-clock timing
    durations = _event_based_timing(kernel_fn, inputs, device, iterations)
    if durations:
        return TimingResult(
            mean_ms=sum(durations) / len(durations),
            durations=durations,
        )

    return TimingResult(mean_ms=0.0)


def profile_reference(
    ref_fn: Callable,
    inputs: tuple,
    *,
    npu_id: int = 0,
    warmup_runs: int = 10,
    iterations: int = 50,
) -> TimingResult:
    """Profile a reference implementation (same as profile_kernel)."""
    return profile_kernel(
        ref_fn, inputs, npu_id=npu_id,
        warmup_runs=warmup_runs, iterations=iterations,
    )
