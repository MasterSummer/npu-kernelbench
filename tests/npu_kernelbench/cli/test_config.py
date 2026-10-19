"""Tests for BenchmarkConfig."""
import pytest
from pydantic import ValidationError
from npu_kernelbench.cli.config import BenchmarkConfig


class TestBenchmarkConfig:
    def test_default_values(self):
        cfg = BenchmarkConfig()
        assert cfg.warmup_runs == 5
        assert cfg.iterations == 10
        assert cfg.timeout == 600
        assert cfg.compile_timeout == 120
        assert cfg.benchmark_reference is True
        assert cfg.npu_id == 0

    def test_custom_values(self):
        cfg = BenchmarkConfig(
            warmup_runs=5, iterations=100, timeout=300, npu_id=1
        )
        assert cfg.warmup_runs == 5
        assert cfg.iterations == 100

    @pytest.mark.parametrize("field,value", [
        ("warmup_runs", -1),
        ("iterations", 0),
        ("iterations", -1),
        ("timeout", 0),
        ("compile_timeout", 0),
    ])
    def test_invalid_values_raise_error(self, field, value):
        with pytest.raises(ValidationError):
            BenchmarkConfig(**{field: value})
