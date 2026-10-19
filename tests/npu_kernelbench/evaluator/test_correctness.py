"""Tests for correctness checking."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from npu_kernelbench.data.trace import Correctness
from npu_kernelbench.data.workload import ToleranceSpec


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCheckTensorSanity:
    def test_no_issues(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.1, 3.0])
        result = check_tensor_sanity(x, y)
        assert result is None

    def test_nan_detected(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("nan"), 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        result = check_tensor_sanity(x, y)
        assert result is not None
        assert result.has_nan is True

    def test_inf_detected(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("inf"), 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        result = check_tensor_sanity(x, y)
        assert result is not None
        assert result.has_inf is True

    def test_all_zero_output(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.zeros(10)
        y = torch.randn(10)
        result = check_tensor_sanity(x, y)
        assert result is not None

    def test_allow_negative_inf(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("-inf")])
        y = torch.tensor([1.0, float("-inf")])
        result = check_tensor_sanity(x, y, allow_negative_inf=True)
        assert result is None

    def test_negative_inf_mismatch_fails(self):
        from npu_kernelbench.evaluator.correctness import check_tensor_sanity

        x = torch.tensor([1.0, float("-inf")])
        y = torch.tensor([1.0, 2.0])
        result = check_tensor_sanity(x, y, allow_negative_inf=True)
        assert result is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestComputeErrorStats:
    def test_exact_match(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False
        assert correctness.max_absolute_error == 0.0
        assert correctness.max_relative_error == 0.0

    def test_small_error_within_tolerance(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec(max_atol=1e-2, max_rtol=1e-2)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.001, 2.001, 3.001])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is False

    def test_large_error_exceeds_tolerance(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec(max_atol=1e-5, max_rtol=1e-5)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([100.0, 200.0, 300.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True

    def test_max_error_cap(self):
        from npu_kernelbench.evaluator.correctness import compute_error_stats

        tol = ToleranceSpec(max_atol=1.0, max_rtol=1.0, max_error_cap=0.5)
        x = torch.tensor([0.0])
        y = torch.tensor([1.0])
        correctness, exceeds = compute_error_stats(x, y, tol)
        assert exceeds is True
