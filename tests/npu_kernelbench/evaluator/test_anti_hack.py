"""Tests for HackDetector AST scanning and kernel signature check."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from npu_kernelbench.evaluator.anti_hack import (
    HackDetector,
    run_ast_scan,
    detect_kernel_signature_mismatch,
    extract_kernel_signatures,
    detect_output_caching,
    detect_monkey_patch,
    detect_clock_anomaly,
    RewardHackResult,
)


# ============================================================
# Reward-hack detector unit tests
# ============================================================


class TestDetectOutputCaching:
    def test_no_caching_varies_with_input(self):
        """Kernel that depends on input should produce all unique outputs."""
        def kernel(x):
            return x + 1

        def input_generator(seed):
            import torch
            return (torch.tensor([seed], dtype=torch.float32),)

        result = detect_output_caching(kernel, input_generator, num_runs=5)
        assert result.detected is False
        assert result.hack_type == "output_caching"
        assert result.evidence["unique_outputs"] == 5

    def test_caching_detected_when_outputs_identical(self):
        """Kernel that ignores input and returns constant triggers detection."""
        import torch

        def kernel(x):
            return torch.tensor([42.0], dtype=torch.float32)

        def input_generator(seed):
            import torch
            return (torch.tensor([seed], dtype=torch.float32),)

        result = detect_output_caching(kernel, input_generator, num_runs=5)
        assert result.detected is True
        assert result.hack_type == "output_caching"
        assert result.evidence["unique_outputs"] < 5


class TestDetectMonkeyPatch:
    def test_matching_reference(self):
        code = "def run(x):\n    return x + 1\n"
        expected_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        result = detect_monkey_patch(code, expected_hash)
        assert result.detected is False
        assert result.hack_type == "monkey_patch"

    def test_modified_reference(self):
        original = "def run(x):\n    return x + 1\n"
        tampered = "def run(x):\n    return x + 2\n"
        expected_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
        result = detect_monkey_patch(tampered, expected_hash)
        assert result.detected is True
        assert result.hack_type == "monkey_patch"


class TestDetectClockAnomaly:
    def test_stable_durations_pass(self):
        """Low-variation durations should not trigger anomaly."""
        durations = [1.0 + i * 0.01 for i in range(20)]
        result = detect_clock_anomaly(durations, max_cv=1.0)
        assert result.detected is False
        assert result.hack_type == "clock_anomaly"

    def test_bimodal_durations_detected(self):
        """High-variation durations should trigger anomaly."""
        durations = [1.0] * 5 + [100.0] * 5 + [1.0] * 5 + [100.0] * 5
        result = detect_clock_anomaly(durations, max_cv=1.0)
        assert result.detected is True
        assert result.hack_type == "clock_anomaly"
        assert result.evidence["cv"] > 1.0

    def test_insufficient_samples(self):
        """Too few samples skip detection gracefully."""
        result = detect_clock_anomaly([1.0, 1.1])
        assert result.detected is False
        assert result.evidence["reason"] == "insufficient samples (need >=10)"


# ============================================================
# HackDetector unit tests
# ============================================================


class TestHackDetector:
    """Direct AST visitor tests (no file I/O)."""

    def _detect(self, code: str) -> list[str]:
        tree = ast.parse(code)
        detector = HackDetector()
        detector.visit(tree)
        return detector.violations

    def test_clean_code(self):
        code = """
import torch
x = torch.zeros(4, 4)
y = torch.randn(4, 4)
z = x + y
"""
        assert self._detect(code) == []

    def test_forbidden_import(self):
        code = "import ctypes"
        violations = self._detect(code)
        assert any("ctypes" in v for v in violations)

    def test_forbidden_from_import(self):
        code = "from ctypes import CDLL"
        violations = self._detect(code)
        assert any("ctypes" in v for v in violations)

    def test_forbidden_torch_api(self):
        code = "torch.matmul(a, b)"
        violations = self._detect(code)
        assert any("torch.matmul" in v for v in violations)

    def test_allowed_torch_api(self):
        code = "torch.zeros(4, 4)"
        assert self._detect(code) == []

    def test_forbidden_torchnpu_api(self):
        code = "torch_npu.some_fake_api()"
        violations = self._detect(code)
        assert any("torch_npu.some_fake_api" in v for v in violations)

    def test_allowed_torchnpu_api(self):
        code = "torch_npu.synchronize()"
        assert self._detect(code) == []

    def test_torch_npu_current_device_allowed(self):
        code = "torch.npu.current_device()"
        assert self._detect(code) == []

    def test_exec_call(self):
        code = 'exec("import ctypes")'
        violations = self._detect(code)
        assert any("exec()" in v for v in violations)

    def test_eval_call(self):
        code = 'eval("1+1")'
        violations = self._detect(code)
        assert any("eval()" in v for v in violations)

    def test_getattr_bypass(self):
        code = 'getattr(torch, "matmul")'
        violations = self._detect(code)
        assert any("getattr" in v for v in violations)
        assert any("matmul" in v for v in violations)

    def test_alias_tracking(self):
        code = """
import torch as th
th.matmul(a, b)
"""
        violations = self._detect(code)
        # Alias th->torch is resolved, so violation mentions torch.matmul
        assert any("torch.matmul" in v for v in violations)

    def test_dynamic_loading_ctypes_cdll(self):
        code = "ctypes.CDLL('lib.so')"
        violations = self._detect(code)
        assert any("ctypes.CDLL" in v for v in violations)

    def test_torch_ops_load_library(self):
        code = 'torch.ops.load_library("lib.so")'
        violations = self._detect(code)
        assert any("torch.ops.load_library" in v for v in violations)

    @pytest.mark.parametrize("imp", ["vllm", "cupy", "cublas", "cudnn", "cufft"])
    def test_forbidden_imports_parametrized(self, imp: str):
        code = f"import {imp}"
        violations = self._detect(code)
        assert any(imp in v for v in violations), f"Expected {imp} to be flagged"

    def test_importlib_import_module(self):
        code = 'importlib.import_module("vllm")'
        violations = self._detect(code)
        assert any("importlib.import_module" in v for v in violations)

    def test_triton_allowed(self):
        code = """
import triton
import triton.language as tl
triton.cdiv(4, 2)
tl.load(x_ptr + offsets)
"""
        violations = self._detect(code)
        assert violations == [], f"Expected no violations, got: {violations}"

    def test_forbidden_attribute_access(self):
        code = """
import ctypes
ctypes.CDLL
"""
        violations = self._detect(code)
        assert any("ctypes" in v for v in violations)

    def test_builtins_exec(self):
        code = 'builtins.exec("import ctypes")'
        violations = self._detect(code)
        assert any("exec()" in v for v in violations)

    def test_builtins_eval(self):
        code = 'builtins.eval("1+1")'
        violations = self._detect(code)
        assert any("eval()" in v for v in violations)

    def test_builtins_import(self):
        code = 'builtins.__import__("vllm")'
        violations = self._detect(code)
        assert any("__import__" in v for v in violations)

    def test_ctypes_cdll_no_double_report(self):
        code = "ctypes.CDLL('lib.so')"
        violations = self._detect(code)
        assert len(violations) == 1
        assert any("ctypes.CDLL" in v for v in violations)

    def test_torch_ops_load_library_no_double_report(self):
        code = 'torch.ops.load_library("lib.so")'
        violations = self._detect(code)
        assert len(violations) == 1
        assert any("torch.ops.load_library" in v for v in violations)

    def test_torch_ops_custom_call_allowed(self):
        """AscendC solutions call torch.ops.xxx.yyy() — must be allowed."""
        code = "torch.ops.cpp_extension_acs.ascendc_add(a, b)"
        assert self._detect(code) == []

    def test_torch_ops_aten_forbidden(self):
        """torch.ops.aten.* is in FORBIDDEN_IMPORTS — must be flagged."""
        code = "torch.ops.aten.add(a, b)"
        violations = self._detect(code)
        assert any("torch.ops.aten" in v for v in violations)

    def test_torch_npu_rotary_mul_allowed(self):
        """Ascend custom ops like torch_npu.npu_rotary_mul must be allowed."""
        code = "torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode='half')"
        assert self._detect(code) == []

    def test_importlib_alias_import_module(self):
        code = """
import importlib as il
il.import_module("vllm")
"""
        violations = self._detect(code)
        assert any("importlib.import_module" in v for v in violations)


# ============================================================
# run_ast_scan integration tests
# ============================================================


class TestRunAstScan:
    def test_empty_list(self):
        assert run_ast_scan([]) == []

    def test_nonexistent_file(self):
        assert run_ast_scan([Path("/nonexistent/file.py")]) == []

    def test_non_py_file(self, tmp_path):
        p = tmp_path / "test_anti_hack.txt"
        p.write_text("import ctypes")
        assert run_ast_scan([p]) == []

    def test_clean_file(self, tmp_path):
        p = tmp_path / "kernel.py"
        p.write_text("import torch\nx = torch.zeros(4, 4)\n")
        assert run_ast_scan([p]) == []

    def test_dirty_file(self, tmp_path):
        p = tmp_path / "kernel.py"
        p.write_text("import ctypes\n")
        violations = run_ast_scan([p])
        assert len(violations) > 0
        assert any("ctypes" in v for v in violations)

    def test_multiple_files(self, tmp_path):
        clean = tmp_path / "clean.py"
        dirty = tmp_path / "dirty.py"
        clean.write_text("x=1")
        dirty.write_text("import ctypes")
        violations = run_ast_scan([clean, dirty])
        assert any("dirty.py" in v for v in violations)


# ============================================================
# extract_kernel_signatures tests
# ============================================================


class TestExtractKernelSignatures:
    def test_nonexistent_dir(self, tmp_path):
        assert extract_kernel_signatures(tmp_path / "nonexistent") == set()

    def test_empty_dir(self, tmp_path):
        assert extract_kernel_signatures(tmp_path) == set()

    def test_python_decorated_function(self, tmp_path):
        p = tmp_path / "kernel.py"
        p.write_text(
            "import triton\n"
            "@triton.jit\n"
            "def matmul_kernel(a, b):\n"
            "    pass\n"
            "def matmul(a, b):\n"  # wrapper, no decorator
            "    return matmul_kernel(a, b)\n"
        )
        assert extract_kernel_signatures(tmp_path) == {"matmul_kernel"}

    def test_python_no_decorator(self, tmp_path):
        p = tmp_path / "kernel.py"
        p.write_text("def run(a, b):\n    return torch.mm(a, b)\n")
        assert extract_kernel_signatures(tmp_path) == set()

    def test_ascendc_global_function(self, tmp_path):
        p = tmp_path / "add_custom.asc"
        p.write_text(
            '__global__ __vector__ void add_custom(GM_ADDR x, GM_ADDR y) {}\n'
        )
        assert extract_kernel_signatures(tmp_path) == {"add_custom"}

    def test_ascendc_aicore_method_not_matched(self, tmp_path):
        p = tmp_path / "kernel.cpp"
        p.write_text(
            'class KernelAdd {\n'
            '    __aicore__ inline void Init() {}\n'
            '};\n'
        )
        assert extract_kernel_signatures(tmp_path) == set()

    def test_ascendc_extern_c(self, tmp_path):
        p = tmp_path / "kernel.cpp"
        p.write_text(
            'extern "C" __global__ __aicore__ void my_kernel(uint32_t n) {}\n'
        )
        assert extract_kernel_signatures(tmp_path) == {"my_kernel"}

    def test_mixed_sources(self, tmp_path):
        (tmp_path / "kernel.py").write_text(
            "@triton.jit\ndef triton_kernel():\n    pass\n"
        )
        (tmp_path / "op.cpp").write_text(
            '__global__ __aicore__ void ascendc_kernel() {}\n'
        )
        assert extract_kernel_signatures(tmp_path) == {
            "triton_kernel", "ascendc_kernel"
        }


# ============================================================
# detect_kernel_signature_mismatch tests
# ============================================================


class TestDetectKernelSignatureMismatch:
    def _make_source(self, tmp_path, kernel_name, lang="python"):
        """Create a source dir with a kernel definition."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        if lang == "python":
            (src_dir / "kernel.py").write_text(
                f"@triton.jit\ndef {kernel_name}():\n    pass\n"
            )
        else:
            (src_dir / "kernel.cpp").write_text(
                f'__global__ __aicore__ void {kernel_name}() {{}}\n'
            )
        return str(src_dir)

    def _make_csv(self, tmp_path, names):
        """Create a trace dir with kernel_details.csv."""
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        lines = ["Name,Start,Duration"]
        for n in names:
            lines.append(f"{n},100,1500")
        (trace_dir / "kernel_details.csv").write_text("\n".join(lines) + "\n")
        return str(trace_dir)

    def test_empty_source_dir(self, tmp_path):
        assert detect_kernel_signature_mismatch("", str(tmp_path)) == []

    def test_empty_trace_dir(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        assert detect_kernel_signature_mismatch(src, "") == []

    def test_nonexistent_source_dir(self, tmp_path):
        assert detect_kernel_signature_mismatch("/nonexistent", str(tmp_path)) == []

    def test_nonexistent_trace_dir(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        assert detect_kernel_signature_mismatch(src, "/nonexistent") == []

    def test_no_kernel_in_source(self, tmp_path):
        """Source has no kernel definition → L2 skipped."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def run(a):\n    return a\n")
        trace = self._make_csv(tmp_path, ["my_kernel"])
        assert detect_kernel_signature_mismatch(str(src_dir), trace) == []

    def test_no_csv_files(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        empty_trace = tmp_path / "trace"
        empty_trace.mkdir()
        assert detect_kernel_signature_mismatch(src, str(empty_trace)) == []

    def test_empty_csv(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "kernel_details.csv").write_text("Name,Start,Duration\n")
        assert detect_kernel_signature_mismatch(src, str(trace_dir)) == []

    def test_matching_kernel(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        trace = self._make_csv(tmp_path, ["my_kernel"])
        assert detect_kernel_signature_mismatch(src, trace) == []

    def test_mismatched_kernel(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        trace = self._make_csv(tmp_path, ["other_kernel"])
        result = detect_kernel_signature_mismatch(src, trace)
        assert len(result) == 1
        assert "signature mismatch" in result[0]
        assert "my_kernel" in result[0]
        assert "other_kernel" in result[0]

    def test_mangled_name_matching(self, tmp_path):
        """C++ mangled name should match via substring."""
        src = self._make_source(tmp_path, "add_custom", lang="ascendc")
        trace = self._make_csv(tmp_path, ["_Z10add_customPhS_S_j"])
        assert detect_kernel_signature_mismatch(src, trace) == []

    def test_missing_name_column(self, tmp_path):
        src = self._make_source(tmp_path, "my_kernel")
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "kernel_details.csv").write_text("Something,Else\n1,2\n")
        assert detect_kernel_signature_mismatch(src, str(trace_dir)) == []
