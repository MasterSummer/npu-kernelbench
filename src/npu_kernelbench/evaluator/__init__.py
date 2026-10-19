"""Evaluator subsystem — correctness, performance, and anti-hack detection."""

from npu_kernelbench.evaluator.evaluator import Evaluator, BaselineHandle
from npu_kernelbench.evaluator.default import DefaultEvaluator
from npu_kernelbench.evaluator.registry import EvaluatorRegistry, get_registry
from npu_kernelbench.evaluator.performance import (
    profile_kernel,
    profile_reference,
    TimingResult,
)
from npu_kernelbench.evaluator.correctness import (
    check_tensor_sanity,
    compute_error_stats,
    set_seed,
    as_tensor_sequence,
)
from npu_kernelbench.evaluator.anti_hack import (
    HackDetector,
    run_ast_scan,
    detect_kernel_signature_mismatch,
)

__all__ = [
    "Evaluator",
    "BaselineHandle",
    "DefaultEvaluator",
    "EvaluatorRegistry",
    "get_registry",
    "profile_kernel",
    "profile_reference",
    "TimingResult",
    "check_tensor_sanity",
    "compute_error_stats",
    "set_seed",
    "as_tensor_sequence",
    "HackDetector",
    "run_ast_scan",
    "detect_kernel_signature_mismatch",
]
