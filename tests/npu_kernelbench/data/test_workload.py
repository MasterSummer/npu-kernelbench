"""Tests for Workload, InputSpec, and ToleranceSpec models."""

from typing import Optional

import pytest
from pydantic import ValidationError

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from npu_kernelbench.data.workload import (
    Workload,
    ToleranceSpec,
    RandomInput,
    ScalarInput,
    CustomInput,
    SafetensorsInput,
    InputSpec,
    resolve_tolerance,
    resolve_max_error_cap,
    _ulp_scale_32,
)


class TestRandomInput:
    def test_default_type(self):
        ri = RandomInput()
        assert ri.type == "random"

    def test_default_shape_is_none(self):
        ri = RandomInput()
        assert ri.shape is None

    def test_with_concrete_shape(self):
        ri = RandomInput(shape=[1, 128, 1024])
        assert ri.shape == [1, 128, 1024]

    def test_with_shape_and_dtype(self):
        ri = RandomInput(shape=[2, 256, 2048], dtype="float16")
        assert ri.shape == [2, 256, 2048]
        assert ri.dtype == "float16"

    def test_with_range(self):
        ri = RandomInput(shape=[32], dtype="int64", range=[0, 127])
        assert ri.shape == [32]
        assert ri.dtype == "int64"
        assert ri.range == [0, 127]

    def test_default_range_is_none(self):
        ri = RandomInput()
        assert ri.range is None

    def test_distribution_default_none(self):
        ri = RandomInput()
        assert ri.distribution is None

    def test_distribution_parsed(self):
        ri = RandomInput(distribution="mixed")
        assert ri.distribution == "mixed"
        ri = RandomInput(distribution="uniform_signed")
        assert ri.distribution == "uniform_signed"

    def test_distribution_invalid_raises(self):
        with pytest.raises(ValidationError):
            RandomInput(distribution="triangular")

    def test_distribution_parameters(self):
        ri = RandomInput(
            distribution="normal",
            mu_low=-3.0,
            mu_high=3.0,
            sigma_low=0.5,
            sigma_high=1.5,
            low=-1.0,
            high=2.0,
            fill="nan",
        )
        assert ri.mu_low == -3.0
        assert ri.mu_high == 3.0
        assert ri.sigma_low == 0.5
        assert ri.sigma_high == 1.5
        assert ri.low == -1.0
        assert ri.high == 2.0
        assert ri.fill == "nan"


class TestScalarInput:
    def test_int_scalar(self):
        si = ScalarInput(value=42)
        assert si.type == "scalar"
        assert si.value == 42

    def test_float_scalar(self):
        si = ScalarInput(value=3.14)
        assert si.type == "scalar"
        assert si.value == 3.14

    def test_bool_scalar(self):
        si = ScalarInput(value=True)
        assert si.type == "scalar"
        assert si.value is True

    def test_list_scalar(self):
        si = ScalarInput(value=[4096])
        assert si.type == "scalar"
        assert si.value == [4096]

    def test_str_scalar(self):
        si = ScalarInput(value="none")
        assert si.type == "scalar"
        assert si.value == "none"


class TestCustomInput:
    def test_default_type(self):
        ci = CustomInput()
        assert ci.type == "custom"


class TestSafetensorsInput:
    def test_valid(self):
        si = SafetensorsInput(path="data.safetensors", tensor_key="weight")
        assert si.type == "safetensors"
        assert si.path == "data.safetensors"
        assert si.tensor_key == "weight"


class TestToleranceSpec:
    def test_defaults(self):
        ts = ToleranceSpec()
        # None fields mean "use the dtype-based precision standard" (resolved
        # during evaluation via resolve_tolerance()).
        assert ts.max_atol is None
        assert ts.max_rtol is None
        assert ts.required_matched_ratio is None
        assert ts.max_error_cap is None
        assert ts.allow_negative_inf is False

    def test_custom(self):
        ts = ToleranceSpec(max_atol=1e-5, max_rtol=1e-4, max_error_cap=0.1)
        assert ts.max_atol == 1e-5
        assert ts.max_error_cap == 0.1


class TestResolveTolerance:
    def test_fills_float16_standard(self):
        ts = resolve_tolerance(ToleranceSpec(), "float16")
        assert ts.max_atol == 2**-9
        assert ts.max_rtol == 2**-9
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap == 1e-1

    def test_fills_float32_standard(self):
        ts = resolve_tolerance(ToleranceSpec(), "float32")
        assert ts.max_atol == 2**-16
        assert ts.max_rtol == 2**-10
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap == 1e-2

    def test_fills_bfloat16_standard(self):
        ts = resolve_tolerance(ToleranceSpec(), "bfloat16")
        assert ts.max_atol == 2**-6
        assert ts.max_rtol == 2**-6
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap == 1e-0

    def test_fills_float8_e4m3fn_standard(self):
        ts = resolve_tolerance(ToleranceSpec(), "float8_e4m3fn")
        assert ts.max_atol == 2**-4
        assert ts.max_rtol == 2**-2
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap == 1e-0

    def test_fills_float8_e5m2_standard(self):
        ts = resolve_tolerance(ToleranceSpec(), "float8_e5m2")
        assert ts.max_atol == 2**-3
        assert ts.max_rtol == 2**-1
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap == 1e-1

    def test_fills_hifloat32_standard(self):
        ts = resolve_tolerance(ToleranceSpec(), "hifloat32")
        assert ts.max_atol == 2**-10
        assert ts.max_rtol == 2**-9
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap == 1e-1

    def test_unknown_dtype_falls_back_to_legacy_defaults(self):
        ts = resolve_tolerance(ToleranceSpec(), "int32")
        assert ts.max_atol == 1e-2
        assert ts.max_rtol == 1e-2
        assert ts.required_matched_ratio == 0.99
        assert ts.max_error_cap is None

    def test_explicit_fields_are_preserved(self):
        ts = resolve_tolerance(
            ToleranceSpec(max_atol=1e-5, max_rtol=1e-4, max_error_cap=0.1), "float32"
        )
        assert ts.max_atol == 1e-5
        assert ts.max_rtol == 1e-4
        assert ts.max_error_cap == 0.1
        assert ts.required_matched_ratio == 0.99


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestUlpScale32:
    """Core 32*ULP computation (opbase experimental_standard)."""

    def test_fp16_large_value(self):
        # golden max = 1024.0 -> exponent=10, fp16 mantissa=10 -> ULP=2^0, 32*ULP=32
        golden = torch.tensor([1.0, 1024.0], dtype=torch.float32)
        assert _ulp_scale_32("float16", golden) == pytest.approx(32.0)

    def test_fp16_unit_value(self):
        # golden max = 1.0 -> exponent=0 -> ULP=2^-10, 32*ULP=2^-5
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert _ulp_scale_32("float16", golden) == pytest.approx(32.0 * 2**-10)

    def test_bf16_value(self):
        # golden max = 2.0 -> exponent=1, bf16 mantissa=7 -> ULP=2^-6
        golden = torch.tensor([2.0], dtype=torch.float32)
        assert _ulp_scale_32("bfloat16", golden) == pytest.approx(32.0 * 2**-6)

    def test_fp32_value(self):
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert _ulp_scale_32("float32", golden) == pytest.approx(32.0 * 2**-23)

    def test_accepts_torch_dtype(self):
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert _ulp_scale_32(torch.float16, golden) == pytest.approx(32.0 * 2**-10)

    def test_unknown_dtype_returns_zero(self):
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert _ulp_scale_32("int32", golden) == 0.0

    def test_zero_golden_returns_zero(self):
        golden = torch.zeros(4, dtype=torch.float32)
        assert _ulp_scale_32("float16", golden) == 0.0

    def test_empty_golden_returns_zero(self):
        golden = torch.empty(0, dtype=torch.float32)
        assert _ulp_scale_32("float16", golden) == 0.0

    def test_infinite_golden_returns_zero(self):
        golden = torch.tensor([float("inf")], dtype=torch.float32)
        assert _ulp_scale_32("float16", golden) == 0.0

    def test_nan_golden_returns_zero(self):
        golden = torch.tensor([float("nan")], dtype=torch.float32)
        assert _ulp_scale_32("float16", golden) == 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestResolveMaxErrorCap:
    """max_abs_error_limit = max(fixed cap, 32*ULP)."""

    def test_ulp_wins_for_large_value(self):
        # 32*ULP(fp16 @1024)=32 > fixed cap 1e-1
        tol = resolve_tolerance(ToleranceSpec(), "float16")
        golden = torch.tensor([1024.0], dtype=torch.float32)
        assert resolve_max_error_cap(tol, "float16", golden) == pytest.approx(32.0)

    def test_fixed_cap_wins_for_small_value(self):
        # 32*ULP(fp16 @1)=2^-5 < fixed cap 1e-1
        tol = resolve_tolerance(ToleranceSpec(), "float16")
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert resolve_max_error_cap(tol, "float16", golden) == pytest.approx(1e-1)

    def test_no_cap_returns_none(self):
        # legacy dtype (int32) resolves to cap=None -> no hard limit
        tol = resolve_tolerance(ToleranceSpec(), "int32")
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert resolve_max_error_cap(tol, "int32", golden) is None

    def test_explicit_cap_is_preserved(self):
        tol = resolve_tolerance(ToleranceSpec(max_error_cap=0.5), "float16")
        golden = torch.tensor([1.0], dtype=torch.float32)
        assert resolve_max_error_cap(tol, "float16", golden) == pytest.approx(0.5)


class TestWorkload:
    def test_valid_workload(self):
        w = Workload(
            axes={"N": 32, "D": 128},
            inputs={
                "x": RandomInput(),
                "alpha": ScalarInput(value=1.0),
            },
            uuid="wl-001",
        )
        assert w.axes == {"N": 32, "D": 128}
        assert w.uuid == "wl-001"

    def test_custom_and_non_custom_mixed_rejected(self):
        with pytest.raises(ValidationError):
            Workload(
                axes={"N": 32},
                inputs={
                    "x": RandomInput(),
                    "y": CustomInput(),
                },
                uuid="wl-002",
            )

    def test_all_custom_inputs_valid(self):
        w = Workload(
            axes={"N": 32},
            inputs={"x": CustomInput(), "y": CustomInput()},
            uuid="wl-003",
        )
        assert isinstance(w.inputs["x"], CustomInput)

    def test_distribution_on_non_random_input_rejected(self):
        # pydantic ignores unknown extra fields on non-RandomInput specs, so
        # a stray "distribution" on e.g. ScalarInput is dropped silently. The
        # validator guards types that natively carry a distribution attribute.
        class FakeInput(ScalarInput):
            distribution: Optional[str] = None

        with pytest.raises(ValidationError, match="not a RandomInput"):
            Workload(
                axes={"N": 32},
                inputs={"x": FakeInput(value=1.0, distribution="zeros")},
                uuid="wl-005",
            )

    def test_random_input_with_distribution_valid(self):
        w = Workload(
            axes={"N": 32},
            inputs={"x": RandomInput(distribution="mixed")},
            uuid="wl-006",
        )
        assert w.inputs["x"].distribution == "mixed"

    def test_get_scalar_inputs(self):
        w = Workload(
            axes={"N": 32},
            inputs={
                "x": RandomInput(),
                "mode": ScalarInput(value="tanh"),
            },
            uuid="wl-004",
        )
        scalars = w.get_scalar_inputs()
        assert scalars == {"mode": "tanh"}

    def test_uuid_is_first_field(self):
        w = Workload(
            uuid="wl-first",
            axes={"N": 32},
            inputs={"x": RandomInput()},
        )
        # uuid should be the first field in model_fields
        fields = list(Workload.model_fields.keys())
        assert fields[0] == "uuid"

    def test_axes_can_be_none(self):
        w = Workload(
            uuid="wl-no-axes",
            axes=None,
            inputs={"x": RandomInput(shape=[1, 128], dtype="float16")},
        )
        assert w.axes is None

    def test_npu_style_workload_no_axes(self):
        w = Workload(
            uuid="fa-001",
            inputs={
                "query": RandomInput(shape=[1, 128, 1024], dtype="float16"),
                "key": RandomInput(shape=[1, 128, 1024], dtype="float16"),
            },
        )
        assert w.uuid == "fa-001"
        assert w.axes is None
        assert w.inputs["query"].shape == [1, 128, 1024]
        assert w.inputs["query"].dtype == "float16"
