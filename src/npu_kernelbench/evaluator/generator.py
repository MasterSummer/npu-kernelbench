"""Input tensor generation for benchmark workloads.

Generates concrete torch.Tensor inputs from Definition + Workload specs,
respecting InputSpec types: RandomInput, ScalarInput, SafetensorsInput, CustomInput.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.dtypes import DType, dtype_str_to_torch_dtype
from npu_kernelbench.data.workload import (
    CustomInput,
    SafetensorsInput,
    ScalarInput,
    TensorListInput,
    Workload,
)


def _rand_tensor(
    shape: list[int],
    dtype: torch.dtype,
    device: torch.device,
    range_spec: Optional[list[int]] = None,
    distribution: Optional[str] = None,
    **dist_kwargs: Any,
) -> torch.Tensor:
    """Generate a random tensor with dtype-aware distribution.

    The default distribution for floating-point types is ``mixed``
    (50% uniform[-5,5] + 50% normal with mu~U[-5,5], sigma~U[0.1,2]).
    Override per-tensor by setting ``distribution`` in the workload JSON.

    Args:
        shape: Tensor dimensions.
        dtype: Target torch dtype.
        device: Target device.
        range_spec: Optional [low, high] for integer types (inclusive).
        distribution: ``"uniform"``, ``"randn"``, ``"mixed"``,
            ``"normal"``, ``"zeros"``, ``"ones"``, ``"uniform_signed"``,
            or ``None`` (defaults to ``"mixed"``).
            * "uniform" — uniform in [low, high] (``dist_kwargs``) or [-5,5].
            * "randn"   — standard normal ``N(0,1)`` (floats only).
            * "normal"  — ``N(mu, sigma)`` with random mu∈[-5,5],
               sigma∈[0.1,2] (floats only).
            * "mixed"   — 50% uniform[-5,5], 50% normal with random
               mu∈[-5,5], sigma∈[0.1,2] (floats only).
            * "zeros"   — all zeros tensor.
            * "ones"    — all ones tensor.
            * "uniform_signed" — uniform(low,high) * random sign (±1).
            * None      — same as ``"mixed"``.
        **dist_kwargs: Extra keyword args for distribution tuning, e.g.
            ``low``, ``high``, ``mu_low``, ``mu_high``, ``sigma_low``,
            ``sigma_high``.

    Returns:
        Random tensor of requested shape and dtype.

    Raises:
        ValueError: If dtype is not supported for random generation.
    """
    fill_value = dist_kwargs.get("fill")
    if fill_value is not None:
        fill_map = {
            "inf": float("inf"),
            "-inf": float("-inf"),
            "nan": float("nan"),
            "zero": 0.0,
        }
        if fill_value not in fill_map:
            raise ValueError(
                f"Unsupported fill value '{fill_value}'. "
                f"Supported: {list(fill_map.keys())}"
            )
        return torch.full(shape, fill_map[fill_value], dtype=dtype, device=device)

    is_float = dtype in (torch.float32, torch.float16, torch.bfloat16)

    if is_float:
        # Default: mixed (50% uniform[-5,5] + 50% normal)
        if distribution is None:
            distribution = "mixed"

        if distribution == "uniform":
            low = float(dist_kwargs.get("low", -5.0))
            high = float(dist_kwargs.get("high", 5.0))
            return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
        if distribution == "zeros":
            return torch.zeros(shape, dtype=dtype, device=device)
        if distribution == "ones":
            return torch.ones(shape, dtype=dtype, device=device)
        if distribution == "uniform_signed":
            low = float(dist_kwargs.get("low", 0.3))
            high = float(dist_kwargs.get("high", 3.0))
            g = torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
            sign = torch.randint(0, 2, shape, device=device).to(dtype) * 2 - 1
            return g * sign

        if distribution in ("normal", "mixed"):
            def _rand_norm():
                mu_low = float(dist_kwargs.get("mu_low", -5.0))
                mu_high = float(dist_kwargs.get("mu_high", 5.0))
                sigma_low = float(dist_kwargs.get("sigma_low", 0.1))
                sigma_high = float(dist_kwargs.get("sigma_high", 2.0))
                mu = float(torch.empty(1, device=device).uniform_(mu_low, mu_high).item())
                sigma = float(torch.empty(1, device=device).uniform_(sigma_low, sigma_high).item())
                # NPU does not support torch.normal for bfloat16, so generate in
                # float32 and cast down to the target dtype.
                if dtype is torch.bfloat16:
                    return torch.normal(mu, sigma, shape, dtype=torch.float32, device=device).to(dtype)
                return torch.normal(mu, sigma, shape, dtype=dtype, device=device)

            if distribution == "mixed":
                # 50% uniform / 50% normal, decided per tensor (statistical 50/50).
                if random.random() < 0.5:
                    return _rand_norm()
                low = float(dist_kwargs.get("low", -5.0))
                high = float(dist_kwargs.get("high", 5.0))
                return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
            return _rand_norm()

        # randn 和任何未识别的 distribution 默认走标准正态
        return torch.randn(shape, dtype=dtype, device=device)

    # Low-precision floats (FP8, FP4) — generate in f32 then clamp and cast
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    ):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)

    # Boolean
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # Integers — range_spec is [low, high) exclusive (matches torch.randint convention)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if range_spec is not None:
            low, high = range_spec
        else:
            ranges: dict[torch.dtype, tuple[int, int]] = {
                torch.int8: (-128, 128),
                torch.int16: (-1024, 1024),
                torch.int32: (-1024, 1024),
                torch.int64: (-1024, 1024),
            }
            low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


def load_safetensors(
    definition: Definition,
    workload: Workload,
) -> dict[str, torch.Tensor]:
    """Load safetensors inputs for a workload.

    Iterates workload.inputs, loads each SafetensorsInput file, validates
    shape and dtype against the definition, and returns CPU tensors.

    Args:
        definition: Kernel definition with input tensor specs.
        workload: Workload with InputSpec entries.

    Returns:
        Dict mapping input name to CPU tensor (pinned memory when available).

    Raises:
        RuntimeError: If safetensors is not installed.
        ValueError: If shape or dtype validation fails.
    """
    try:
        import safetensors.torch as st
    except ImportError as e:
        raise RuntimeError(
            "safetensors is required for SafetensorsInput workloads. "
            "Install with: pip install safetensors"
        ) from e

    expected_shapes = definition.get_input_shapes(workload.axes)
    safe_tensors: dict[str, torch.Tensor] = {}
    loaded_files: dict[str, dict[str, torch.Tensor]] = {}

    for name, input_spec in workload.inputs.items():
        if input_spec.type != "safetensors":
            continue

        path = str(input_spec.path)

        # Cache loaded files to avoid re-reading the same path
        if path not in loaded_files:
            loaded_files[path] = st.load_file(path)
        tensors = loaded_files[path]

        if input_spec.tensor_key not in tensors:
            raise ValueError(
                f"Missing key '{input_spec.tensor_key}' in '{path}'"
            )

        t = tensors[input_spec.tensor_key]

        # Validate shape
        expected_shape = expected_shapes[name]
        if list(t.shape) != expected_shape:
            raise ValueError(
                f"'{name}' expected shape {expected_shape}, got {list(t.shape)}"
            )

        # Validate dtype
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(
                f"'{name}' expected dtype {expect_dtype}, got {t.dtype}"
            )

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()

        safe_tensors[name] = t

    return safe_tensors


def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    seed: int,
    *,
    safe_tensors: Optional[dict[str, torch.Tensor]] = None,
    custom_inputs_fn: Optional[Callable] = None,
) -> list[Any]:
    """Generate input values for a workload in definition order.

    Iterates definition.inputs in order and dispatches based on the
    corresponding workload.inputs InputSpec type.

    Args:
        definition: Kernel definition with input tensor specs.
        workload: Workload with concrete axis values and InputSpec entries.
        device: Target device string (e.g., "npu:0").
        seed: RNG seed for reproducibility.
        safe_tensors: Pre-loaded safetensors dict from load_safetensors().
        custom_inputs_fn: Optional callable that returns dict[name, Tensor]
            for CustomInput entries. Signature:
            (resolved_axes: dict, device: torch.device) -> dict[str, Tensor].

    Returns:
        List of input values (torch.Tensor or scalar) in definition.inputs order.

    Raises:
        RuntimeError: If a required SafetensorsInput or CustomInput is missing.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

    # Per-case distribution family for float inputs without an explicit
    # "distribution" override: the whole case shares one family (normal or
    # uniform), matching the 50/50 mix required by experimental_standard.
    case_distribution = "normal" if random.random() < 0.5 else "uniform"

    shapes = definition.get_input_shapes(workload.axes)
    scalar_inputs = workload.get_scalar_inputs()
    dev = torch.device(device)
    out: list[Any] = []

    # Pre-compute custom tensors if a factory is provided
    custom_tensors: Optional[dict[str, Any]] = None
    if custom_inputs_fn is not None:
        resolved_axes = definition.get_resolved_axes_values(workload.axes)
        custom_tensors = custom_inputs_fn(resolved_axes, dev)

    for name, spec in definition.inputs.items():
        input_spec = workload.inputs.get(name)

        # --- Handle SafetensorsInput, ScalarInput, CustomInput ---
        if isinstance(input_spec, SafetensorsInput):
            if safe_tensors is None or name not in safe_tensors:
                raise RuntimeError(
                    f"Missing required safetensors input '{name}'. "
                    f"Call load_safetensors() before gen_inputs()."
                )
            t_cpu = safe_tensors[name]
            out.append(t_cpu.to(device=dev, non_blocking=True))

        elif isinstance(input_spec, TensorListInput):
            dtype = dtype_str_to_torch_dtype(input_spec.dtype)
            tensors = [
                _rand_tensor(list(s), dtype, dev, distribution=case_distribution)
                for s in input_spec.shapes
            ]
            out.append(tensors)

        elif name in scalar_inputs:
            out.append(scalar_inputs[name])

        elif isinstance(input_spec, CustomInput):
            if custom_tensors is None or name not in custom_tensors:
                raise RuntimeError(
                    f"CustomInput for '{name}' requires custom_inputs_fn. "
                    f"Definition '{definition.name}' has "
                    f"custom_inputs_entrypoint='{definition.custom_inputs_entrypoint}'."
                )
            val = custom_tensors[name]
            if isinstance(val, torch.Tensor):
                out.append(val.to(device=dev, non_blocking=True))
            else:
                out.append(val)

        else:
            # --- Optional input: not in workload and no dtype/shape in definition ---
            if input_spec is None and spec.dtype is None and spec.shape is None:
                out.append(None)
                continue

            # --- Resolve dtype: RandomInput.dtype -> Definition default -> error ---
            dtype_str = None
            if (
                input_spec is not None
                and input_spec.type == "random"
                and getattr(input_spec, "dtype", None) is not None
            ):
                dtype_str = input_spec.dtype
            elif spec.dtype is not None:
                raw = spec.dtype
                dtype_str = raw.value if isinstance(raw, DType) else raw

            if dtype_str is None:
                raise ValueError(
                    f"Cannot determine dtype for input '{name}': "
                    f"no dtype in definition or workload"
                )
            dtype = dtype_str_to_torch_dtype(dtype_str)

            # --- Resolve shape: RandomInput.shape -> Definition axes -> error ---
            shape = None
            if (
                input_spec is not None
                and input_spec.type == "random"
                and getattr(input_spec, "shape", None) is not None
            ):
                shape = tuple(input_spec.shape)
            elif spec.shape is not None:
                shape = shapes.get(name)
            else:
                raise ValueError(
                    f"Cannot determine shape for input '{name}': "
                    f"no shape in definition or workload"
                )

            range_spec = None
            distribution = None
            dist_kwargs: dict[str, Any] = {}
            if input_spec is not None and input_spec.type == "random":
                if getattr(input_spec, "range", None) is not None:
                    range_spec = input_spec.range
                if getattr(input_spec, "distribution", None) is not None:
                    distribution = input_spec.distribution
                for key in ("low", "high", "mu_low", "mu_high", "sigma_low", "sigma_high"):
                    val = getattr(input_spec, key, None)
                    if val is not None:
                        dist_kwargs[key] = float(val)
                fill = getattr(input_spec, "fill", None)
                if fill is not None:
                    dist_kwargs["fill"] = fill

            # Float inputs without an explicit distribution inherit the
            # case-level family (normal/uniform 50/50, per experimental_standard).
            if distribution is None and dtype in (torch.float32, torch.float16, torch.bfloat16):
                distribution = case_distribution

            if shape is None:
                value = _rand_tensor((), dtype, dev, range_spec, distribution, **dist_kwargs).item()
            else:
                value = _rand_tensor(list(shape), dtype, dev, range_spec, distribution, **dist_kwargs)

            out.append(value)

    return out
