"""Definition model — kernel specification with symbolic axes, tensor specs, and reference code."""

from __future__ import annotations

import ast
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from pydantic import Field, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString
from .dtypes import DType, dtype_str_to_torch_dtype
from .shapes import AxisConst, AxisExpr, AxisVar, resolve_shape_expression

if TYPE_CHECKING:
    import torch

AxisSpec = Union[AxisConst, AxisVar, AxisExpr]
"""Union type representing all possible axis specifications."""


class TensorSpec(BaseModelWithDocstrings):
    """Specification for a tensor including shape and data type."""

    shape: Optional[list[NonEmptyString]] = None
    """List of axis names defining the tensor shape. None for scalar values or NPU-style deferral."""
    dtype: Optional[DType] = None
    """The data type of all elements in this tensor. None for NPU-style deferral to workload."""
    description: Optional[str] = None
    """Optional human-readable description. In NPU-style, carries semantic capability notes."""


class Definition(BaseModelWithDocstrings):
    """Complete definition of a computational workload.

    Provides a formal specification for a kernel: tensor formats, dimension
    semantics, and reference implementation. This is the single source of
    truth for kernel development and optimization.
    """

    name: NonEmptyString
    """A unique, human-readable name for the kernel definition."""
    op_type: Optional[NonEmptyString] = None
    """The general compute category."""
    axes: dict[NonEmptyString, AxisSpec]
    """Dictionary of symbolic dimensions used in tensor shapes."""
    custom_inputs_entrypoint: Optional[NonEmptyString] = None
    """Entrypoint function to generate custom inputs. Signature: (axes, device) -> dict[str, Tensor]."""
    inputs: dict[NonEmptyString, TensorSpec]
    """Named input tensors required by this kernel (ordered)."""
    outputs: dict[NonEmptyString, TensorSpec]
    """Named output tensors produced by this kernel (ordered)."""
    reference: NonEmptyString
    """Reference implementation Python code. Must define a 'run' function."""
    reference_hash: str = ""
    """SHA256 of reference code, populated by the benchmark framework.

    Used by the monkey_patch anti-hack check to detect tampering with the
    reference implementation between problem creation and evaluation.
    """
    description: Optional[str] = None
    """Optional human-readable description."""

    @model_validator(mode="after")
    def _validate_reference_code(self) -> "Definition":
        try:
            mod = ast.parse(self.reference, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Reference must be valid Python code: {e}") from e

        has_run_func = any(
            isinstance(node, ast.FunctionDef) and node.name == "run"
            for node in mod.body
        )
        if not has_run_func:
            raise ValueError("Reference must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> "Definition":
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> "Definition":
        all_tensors = {**self.inputs, **self.outputs}
        for tensor_name, tensor_spec in all_tensors.items():
            if tensor_spec.shape is None:
                continue
            for axis_name in tensor_spec.shape:
                if axis_name.isdigit():
                    continue
                if axis_name not in self.axes:
                    raise ValueError(
                        f'Tensor "{tensor_name}" references undefined axis "{axis_name}".'
                    )
        return self

    @cached_property
    def const_axes(self) -> dict[str, int]:
        return {
            name: axis.value
            for name, axis in self.axes.items()
            if isinstance(axis, AxisConst)
        }

    @cached_property
    def var_axes(self) -> list[str]:
        return [name for name, axis in self.axes.items() if isinstance(axis, AxisVar)]

    @cached_property
    def expr_axes(self) -> dict[str, AxisExpr]:
        return {name: axis for name, axis in self.axes.items() if isinstance(axis, AxisExpr)}

    def get_axes_values(
        self, input_shapes: Iterable[Optional[tuple[int, ...]]]
    ) -> dict[str, int]:
        var_axes_values: dict[str, int] = {}
        for (inp_name, inp_spec), inp_shape in zip(self.inputs.items(), input_shapes):
            if inp_spec.shape is None:
                continue
            if len(inp_spec.shape) != len(inp_shape):
                raise ValueError(
                    f"Input '{inp_name}' defined dim {len(inp_spec.shape)} "
                    f"but actual dim is {len(inp_shape)}"
                )
            for axis_name, axis_value in zip(inp_spec.shape, inp_shape):
                if axis_name in self.axes and self.axes[axis_name].type == "var":
                    if axis_name in var_axes_values:
                        if var_axes_values[axis_name] != axis_value:
                            raise ValueError(
                                f"Axis '{axis_name}' has inconsistent values: "
                                f"{var_axes_values[axis_name]} vs {axis_value}"
                            )
                    else:
                        var_axes_values[axis_name] = axis_value

        if len(var_axes_values) != len(self.var_axes):
            missing = set(self.var_axes) - set(var_axes_values.keys())
            raise ValueError(f"Missing values for variable axes: {missing}")
        return var_axes_values

    def get_resolved_axes_values(self, var_axes_values: dict[str, int]) -> dict[str, int]:
        resolved: dict[str, int] = self.const_axes.copy()
        resolved.update(var_axes_values)
        for name, axis in self.expr_axes.items():
            resolved[name] = resolve_shape_expression(axis.expression, resolved)
        return resolved

    def _get_shapes(
        self,
        tensors: Iterable[TensorSpec],
        var_axes_values: Optional[dict[str, int]] = None,
    ) -> list[Optional[tuple[int, ...]]]:
        var_axes_values = var_axes_values or {}
        resolved_axes = self.get_resolved_axes_values(var_axes_values)
        shapes = []
        for tensor_spec in tensors:
            if tensor_spec.shape is None:
                shapes.append(None)
                continue
            shape = []
            for axis_name in tensor_spec.shape:
                if axis_name.isdigit():
                    value = int(axis_name)
                else:
                    value = resolved_axes[axis_name]
                shape.append(value)
            shapes.append(tuple(shape))
        return shapes

    def get_input_shapes(
        self, var_axes_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        shapes = self._get_shapes(self.inputs.values(), var_axes_values)
        return dict(zip(self.inputs.keys(), shapes))

    def get_output_shapes(
        self, var_axes_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        shapes = self._get_shapes(self.outputs.values(), var_axes_values)
        return dict(zip(self.outputs.keys(), shapes))

    @cached_property
    def torch_input_dtypes(self) -> list["torch.dtype"]:
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.inputs.values()]

    @cached_property
    def torch_output_dtypes(self) -> list["torch.dtype"]:
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.outputs.values()]
