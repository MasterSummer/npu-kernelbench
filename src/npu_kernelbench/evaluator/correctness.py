"""Numerical correctness checking for kernel evaluation."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from npu_kernelbench.data.trace import Correctness
from npu_kernelbench.data.workload import ToleranceSpec


def as_tensor_sequence(value) -> list:
    """Normalize an output value to a list of tensors.

    A single ``torch.Tensor`` is wrapped in a one-element list; a
    tuple/list of tensors is returned as-is. This lets variadic outputs
    (e.g. ``torch.nonzero(as_tuple=True)`` on n-D input, which returns n
    tensors) be compared element-wise against a fixed-arity definition
    schema that declares a single output.
    """
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output.

    Returns a Correctness describing the failure, or None if ok.
    """
    ref_nonfinite = ~torch.isfinite(ref_tensor)
    sol_nonfinite = ~torch.isfinite(sol_tensor)

    if allow_negative_inf:
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_nonfinite = ref_nonfinite & ~both_neg_inf
        sol_nonfinite = sol_nonfinite & ~both_neg_inf

    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()
    if has_nonfinite:
        has_nan = (
            torch.isnan(sol_tensor).any().item()
            or torch.isnan(ref_tensor).any().item()
        )
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    # All-zero output check
    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec
) -> Tuple[Correctness, bool]:
    """Compute numerical error between output and reference.

    Returns (correctness, exceeds_tolerance).
    """
    # Opbase experimental standard: resolve dtype-based tolerance once here so
    # every caller (DefaultEvaluator, standalone eval.py, ...) gets the same
    # precision standard without duplicating resolve_tolerance calls.
    from npu_kernelbench.data.workload import resolve_tolerance

    tolerance = resolve_tolerance(tolerance, reference.dtype)

    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    allow_neg_inf = tolerance.allow_negative_inf

    infs_nans = check_tensor_sanity(x, y, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return infs_nans, True

    if allow_neg_inf:
        both_neg_inf = (x == float("-inf")) & (y == float("-inf"))
        finite_mask = ~both_neg_inf
        x = x[finite_mask]
        y = y[finite_mask]

    abs_error = torch.abs(x - y)
    total_elements = abs_error.numel()
    if total_elements == 0:
        return Correctness(), False

    max_abs = float(abs_error.max().item())

    tol_bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)
    exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)

    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None:
        # Opbase experimental standard: max_abs_error_limit = max(fixed cap,
        # 32*ULP) — resolved in workload.py so the standard stays defined in
        # one place. ULP scales with the golden's magnitude, so large-value
        # accumulations/reductions are not killed by a fixed absolute cap.
        from npu_kernelbench.data.workload import resolve_max_error_cap

        cap_limit = resolve_max_error_cap(tolerance, reference.dtype, y)
        if cap_limit is not None and max_abs > cap_limit:
            exceeds_tol = True

    rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)
    max_rel = float(rel_error.max().item())

    return Correctness(
        max_absolute_error=max_abs, max_relative_error=max_rel
    ), exceeds_tol
