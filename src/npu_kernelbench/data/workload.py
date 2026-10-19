"""Workload specification — defines input tensors and tolerances for a single benchmark run."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Optional, Union

import torch
from pydantic import Field, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class DistributionType(str, Enum):
    """Distribution strategy for float random tensors."""

    MIXED = "mixed"
    """50% uniform(-5,5) + 50% normal with random mu~U(-5,5) and sigma~U(0.1,2).
    The uniform branch defaults to low=-5, high=5 (overridable via low/high);
    the normal branch samples mu/sigma per case (overridable via
    mu_low/mu_high/sigma_low/sigma_high)."""
    RANDN = "randn"
    """Standard normal N(0,1)."""
    UNIFORM = "uniform"
    """Uniform(low, high), default uniform(-5,5); low/high overridable via low/high."""
    NORMAL = "normal"
    """Normal N(mu, sigma) with random mu~U(-5,5) and sigma~U(0.1,2);
    ranges overridable via mu_low/mu_high/sigma_low/sigma_high."""
    ZEROS = "zeros"
    """All zeros tensor (e.g. m/v in AdamW, bias in GroupNormSwish)."""
    ONES = "ones"
    """All ones tensor (e.g. weight in GroupNormSwish)."""
    UNIFORM_SIGNED = "uniform_signed"
    """Uniform with random sign flip: uniform_(low,high) * rand_sign,
    default low=0.3, high=3.0 (value range [-3,-0.3] U [0.3,3], no near-zero
    values). Used for AdamW grad."""


class RandomInput(BaseModelWithDocstrings):
    """Random input generation descriptor."""

    type: Literal["random"] = "random"
    dtype: Optional[str] = None
    """Override the default dtype from Definition. When None, use Definition's dtype."""
    shape: Optional[list[int]] = None
    """Concrete shape for this tensor. When provided, bypasses symbolic axis resolution."""
    range: Optional[list[int]] = None
    """Range [low, high) for integer random generation (high exclusive)."""
    distribution: Optional[DistributionType] = None
    """Float tensor distribution strategy. None defaults to MIXED (50/50 uniform+normal)."""
    mu_low: Optional[float] = None
    """Lower bound for mu (mean) in normal distribution."""
    mu_high: Optional[float] = None
    """Upper bound for mu (mean) in normal distribution."""
    sigma_low: Optional[float] = None
    """Lower bound for sigma (std) in normal distribution."""
    sigma_high: Optional[float] = None
    """Upper bound for sigma (std) in normal distribution."""
    low: Optional[float] = None
    """Lower bound for uniform distribution."""
    high: Optional[float] = None
    """Upper bound for uniform distribution."""
    fill: Optional[str] = None
    """Fill value: 'inf', '-inf', 'nan', 'zero'. When set, generate torch.full instead of random."""


class ScalarInput(BaseModelWithDocstrings):
    """Scalar literal input specification."""

    type: Literal["scalar"] = "scalar"
    value: Union[int, float, bool, str, list]


class CustomInput(BaseModelWithDocstrings):
    """Custom input generated from reference code entrypoint."""

    type: Literal["custom"] = "custom"


class SafetensorsInput(BaseModelWithDocstrings):
    """Input loaded from safetensors file."""

    type: Literal["safetensors"] = "safetensors"
    path: NonEmptyString
    tensor_key: NonEmptyString


class TensorListInput(BaseModelWithDocstrings):
    """Variable-length list of tensors sharing the same dtype."""

    type: Literal["tensor_list"] = "tensor_list"
    dtype: str
    """Data type shared by all tensors in the list."""
    shapes: list[list[int]]
    """Shape of each tensor in the list."""


InputSpec = Union[RandomInput, SafetensorsInput, ScalarInput, CustomInput, TensorListInput]
"""Union type for all input specification types."""


class ToleranceSpec(BaseModelWithDocstrings):
    """Tolerance specification for correctness checking.

    When fields are None, the dtype-based precision standard is applied
    automatically during evaluation (see resolve_tolerance()).
    """

    max_atol: Optional[float] = Field(default=None)
    """Maximum absolute error allowed. None = use dtype standard."""
    max_rtol: Optional[float] = Field(default=None)
    """Maximum relative error allowed. None = use dtype standard."""
    required_matched_ratio: Optional[float] = Field(default=None)
    """Ratio of elements that must pass correctness bounds. None = use dtype standard (0.99)."""
    max_error_cap: Optional[float] = Field(default=None)
    """Hard ceiling on maximum absolute error. None = use dtype standard."""
    allow_negative_inf: bool = Field(default=False)
    """When True, matching -inf values are treated as correct."""


# ---------------------------------------------------------------------------
# Dtype-based precision standard (aligned with CANN opbase experimental_standard)
# Reference: https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
# ---------------------------------------------------------------------------

DTYPE_TOLERANCE_STANDARDS: dict[str, dict] = {
    "float16": {
        "max_atol": 2**-9,       # ~1.953e-3
        "max_rtol": 2**-9,       # ~1.953e-3
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-1,
    },
    "bfloat16": {
        "max_atol": 2**-6,       # ~1.5625e-2
        "max_rtol": 2**-6,       # ~1.5625e-2
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-0,
    },
    "float32": {
        "max_atol": 2**-16,      # ~1.526e-5
        "max_rtol": 2**-10,      # ~9.766e-4
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-2,
    },
    "float64": {
        "max_atol": 2**-16,      # ~1.526e-5
        "max_rtol": 2**-10,      # ~9.766e-4
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-2,
    },
    "float8_e4m3fn": {
        "max_atol": 2**-4,       # ~6.25e-2
        "max_rtol": 2**-2,       # ~2.5e-1
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-0,
    },
    "float8_e5m2": {
        "max_atol": 2**-3,       # ~1.25e-1
        "max_rtol": 2**-1,       # ~5e-1
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-1,
    },
    "hifloat32": {
        "max_atol": 2**-10,      # ~9.766e-4
        "max_rtol": 2**-9,       # ~1.953e-3
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-1,
    },
}

# Fallback for unrecognized dtypes (use old defaults)
_DEFAULT_TOLERANCE = {
    "max_atol": 1e-2,
    "max_rtol": 1e-2,
    "required_matched_ratio": 0.99,
    "max_error_cap": None,
}


def resolve_tolerance(
    tolerance: ToleranceSpec, dtype: "torch.dtype | str | None" = None
) -> ToleranceSpec:
    """Resolve a ToleranceSpec by filling None fields from the dtype-based standard.

    If a field is explicitly set (not None), it is preserved as-is.
    If dtype is None or unrecognized, falls back to legacy defaults (atol=1e-2, rtol=1e-2).
    """
    # Normalize dtype to string key
    dtype_key: str | None = None
    if dtype is not None:
        dtype_key = str(dtype).replace("torch.", "")

    standard = DTYPE_TOLERANCE_STANDARDS.get(dtype_key, _DEFAULT_TOLERANCE) if dtype_key else _DEFAULT_TOLERANCE

    return ToleranceSpec(
        max_atol=tolerance.max_atol if tolerance.max_atol is not None else standard["max_atol"],
        max_rtol=tolerance.max_rtol if tolerance.max_rtol is not None else standard["max_rtol"],
        required_matched_ratio=(
            tolerance.required_matched_ratio
            if tolerance.required_matched_ratio is not None
            else standard["required_matched_ratio"]
        ),
        max_error_cap=(
            tolerance.max_error_cap
            if tolerance.max_error_cap is not None
            else standard.get("max_error_cap")
        ),
        allow_negative_inf=tolerance.allow_negative_inf,
    )


_MANTISSA_BITS: dict[str, int] = {
    "float16": 10,
    "bfloat16": 7,
    "float32": 23,
    "float64": 52,
    "float8_e4m3fn": 3,
    "float8_e5m2": 2,
    "hifloat32": 11,
}


def _ulp_scale_32(dtype: "torch.dtype | str | None", golden) -> float:
    """Return 32 * ULP at the golden's largest magnitude for ``dtype``.

    ULP (unit in the last place) of a normal float is 2^(floor(log2(|v|)) -
    mantissa_bits). Per the opbase experimental standard, the absolute error
    hard limit is ``max(fixed_cap, 32 * ULP)``, so large-magnitude outputs
    get a proportionally larger allowance instead of being killed by a fixed
    cap. Falls back to 0.0 (i.e. fixed cap only) for unsupported dtypes,
    zero/empty goldens or non-finite maxima.
    """
    dtype_key = str(dtype).replace("torch.", "") if dtype is not None else ""
    mantissa = _MANTISSA_BITS.get(dtype_key)
    if mantissa is None:
        return 0.0
    golden_f = golden.float() if golden.dtype != torch.float32 else golden
    max_abs = float(golden_f.abs().max().item()) if golden_f.numel() else 0.0
    if max_abs <= 0.0 or not (max_abs == max_abs) or max_abs == float("inf"):
        return 0.0
    exponent = math.floor(math.log2(max_abs))  # floor(log2(|v|)) for any |v| > 0
    return 32.0 * (2.0 ** (exponent - mantissa))


def resolve_max_error_cap(
    tolerance: ToleranceSpec, dtype: "torch.dtype | str | None", golden
) -> "float | None":
    """Resolve the absolute error hard limit for a given golden tensor.

    Per the opbase experimental standard: ``max_abs_error_limit = max(fixed
    cap, 32 * ULP)``, where ULP is measured at the golden's largest magnitude.
    The fixed cap comes from the (already resolved) tolerance; None means no
    hard limit (legacy dtypes without a cap entry).
    """
    if tolerance.max_error_cap is None:
        return None
    return max(tolerance.max_error_cap, _ulp_scale_32(dtype, golden))


class Workload(BaseModelWithDocstrings):
    """Concrete workload configuration for benchmarking."""

    uuid: NonEmptyString
    """Unique identifier for this workload configuration."""
    axes: Optional[dict[str, NonNegativeInt]] = None
    """Dictionary mapping axis names to their concrete integer values. None for NPU-style workloads."""
    inputs: dict[str, InputSpec]
    """Dictionary mapping input names to their data specifications."""
    tolerance: ToleranceSpec = Field(default=ToleranceSpec())
    """Tolerance specification for this workload."""

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Workload":
        custom_inputs = [
            name
            for name, inp in self.inputs.items()
            if isinstance(inp, CustomInput)
        ]
        non_custom_inputs = [
            name
            for name, inp in self.inputs.items()
            if not isinstance(inp, CustomInput)
        ]
        if len(custom_inputs) > 0 and len(non_custom_inputs) > 0:
            raise ValueError(
                f"Cannot mix custom and non-custom inputs. "
                f"Custom: {custom_inputs}. Non-custom: {non_custom_inputs}"
            )
        for name, inp in self.inputs.items():
            if not isinstance(inp, RandomInput) and getattr(inp, "distribution", None) is not None:
                raise ValueError(
                    f"Input '{name}' has 'distribution' but is not a RandomInput"
                )
        return self

    def get_scalar_inputs(self) -> dict[str, Union[int, float, bool, str]]:
        return {
            name: inp.value
            for name, inp in self.inputs.items()
            if isinstance(inp, ScalarInput)
        }
