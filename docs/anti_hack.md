# NPUKernelBench 反作弊机制说明

NPUKernelBench 在算子评测流水线中内置多层反作弊检查，用于识别常见的投机/作弊行为，保证评测结果真实反映算子实现的正确性与性能。

---

## 检查层级总览

| 层级 | 名称 | 检查时机 | 目标作弊行为 |
|------|------|----------|--------------|
| L0 | `ast_scan` | 评测启动前（每个 workload 前复用结果） | 源码中出现危险导入、禁用 API、动态加载等 |
| L1 | `kernel_signature` | 性能 profiling 完成后 | 实际执行的 kernel 与源码声明的 kernel 不一致 |
| L2 | `output_caching` | 性能 profiling 完成后 | 不同输入下返回相同输出（结果缓存/硬编码） |
| L2 | `monkey_patch` | 性能 profiling 完成后 | 参考实现被运行时篡改 |
| L2 | `clock_anomaly` | 性能 profiling 完成后 | 计时数据异常波动（人为降速/时钟作弊） |

所有 L2 检查统称为 **Reward-Hack Detection**，统一通过 `RewardHackResult` 返回检测结果。

---

## 启用与关闭

通过 `config.json` 中的 `enable_anti_hack` 字段控制：

```json
{
  "enable_anti_hack": ["ast_scan", "kernel_signature"]
}
```

完整可选值：

- `"ast_scan"`
- `"kernel_signature"`
- `"output_caching"`
- `"monkey_patch"`
- `"clock_anomaly"`

设为 `[]` 即可关闭全部反作弊检查：

```json
{
  "enable_anti_hack": []
}
```

CLI 也支持通过 `--enable-anti-hack` 多次指定：

```bash
uv run npu-kernelbench eval my_operator \
  --solution solution.json \
  --enable-anti-hack ast_scan \
  --enable-anti-hack kernel_signature \
  --enable-anti-hack output_caching
```

> 注意：`enable_anti_hack` 默认值为 `["ast_scan", "kernel_signature"]`，**不包含** reward-hack 检查。如需启用 reward-hack，请显式配置。

---

## L0: AST 静态扫描（`ast_scan`）

### 原理

在 solution 源码被加载前，使用 `ast.NodeVisitor` 遍历 `sources/` 下的所有 `.py` 文件，检查是否包含：

- **禁用导入**：`ctypes`、`vllm`、`cupy`、`cublas`、`cudnn`、`cufft`、`torch.ops.aten`、`torch.ops.vllm` 等
- **危险调用**：`exec()`、`eval()`、`getattr()`、`__import__()`、`importlib.import_module()`
- **禁用 PyTorch API**：`torch.matmul`、`torch.nn.functional.*`、`torch.ops.aten.*` 等
- **危险动态加载**：`ctypes.CDLL`、`torch.ops.load_library`
- **非法 torch_npu API**：不在白名单内的 `torch_npu.*` 调用

### 合法 API 白名单

允许使用的 API 主要包括：

- 张量创建：`torch.empty`、`torch.zeros`、`torch.randn`、`torch.full_like` 等
- 数据类型与常量：`torch.float16`、`torch.int32`、`torch.pi` 等
- 设备/同步：`torch.npu.synchronize`、`torch_npu.synchronize`、`torch_npu.set_device`
- Ascend 自定义算子：`torch_npu.npu_*`（如 `torch_npu.npu_rotary_mul`）
- Triton：`triton`、`triton.language`
- AscendC 注册：`torch.ops.cpp_extension_acs.*`

### 触发效果

AST 扫描发现的违规会作为 `log` 写入 `Trace.evaluation.log`，并将最终 `EvaluationStatus` 标记为 `REWARD_HACK`。

---

## L1: Kernel 签名校验（`kernel_signature`）

### 原理

1. 从 `sources/` 中提取源码里声明的所有 kernel 名称：
   - Python Triton：`@triton.jit` 装饰的函数
   - Ascend C：`__global__ __aicore__ void kernel_name(...)` 或 `__global__ __vector__ void kernel_name(...)`
2. 读取 CANN Profiler 生成的 `kernel_details.csv`，拿到实际在 NPU 上执行的 kernel 名称列表。
3. 对源码中声明的每个 kernel，检查 profiler 中是否至少有一个实际执行名称包含它（支持 C++ 混淆名）。
4. 如果源码声明了 kernel，但 profiler 中没有匹配的执行记录，则判定为签名校验失败。

### 典型作弊场景

- 源码写了一个 dummy `@triton.jit` kernel，实际用 PyTorch 算子完成计算。
- 调用了外部预编译 `.so`，但源码没有对应的 kernel 声明。

### 跳过条件

如果源码中**没有任何** `@triton.jit` 或 `__global__` 定义的 kernel，L1 检查会自动跳过，不做判定。

### 触发效果

签名校验失败会追加一条类似如下的 log：

```
Reward hack (kernel_signature): Kernel signature mismatch: declared kernels {my_kernel} not found in profile traces
```

最终状态标记为 `REWARD_HACK`。

---

## L2: 输出缓存检测（`output_caching`）

### 原理

1. 使用**不同的随机种子**生成 5 组输入。
2. 对每组输入执行 solution 的 `run()` 函数。
3. 比较 5 次输出：如果所有输出完全相同，说明 kernel 可能忽略了输入变化、缓存了上一次的输出或硬编码了结果。

### 触发条件

- 5 次运行输出完全相同（按张量内容进行判等）。

### 触发效果

```
Reward hack (output_caching): Output does not vary across inputs ...
```

最终状态标记为 `REWARD_HACK`。

---

## L2: 参考实现篡改检测（`monkey_patch`）

### 原理

1. 在 `Definition` 构建时计算 `reference` 源码的 SHA256 哈希，存入 `reference_hash` 字段。
2. 评测时，再次对 `definition.reference` 源码计算 SHA256。
3. 如果运行时哈希与 `reference_hash` 不一致，说明参考实现被篡改。

### 依赖

- `Definition.reference_hash` 必须在出题/数据生成时正确填入。
- 如果 `reference_hash` 为空字符串，`monkey_patch` 检查会自动跳过。

### 触发效果

```
Reward hack (monkey_patch): Reference code hash mismatch ...
```

最终状态标记为 `REWARD_HACK`。

---

## L2: 时钟异常检测（`clock_anomaly`）

### 原理

1. 收集 profiling 阶段多次迭代的时间样本 `durations`。
2. 过滤异常值后计算变异系数（Coefficient of Variation，CV）：

   ```
   CV = std(durations) / mean(durations)
   ```

3. 如果 CV 超过阈值（默认 `1.0`），判定为计时数据异常波动。

### 典型作弊场景

- 某些实现通过人为制造延迟波动干扰性能评测。
- 计时样本中存在明显的双峰/多峰分布。

### 触发效果

```
Reward hack (clock_anomaly): Clock anomaly detected (CV=2.34) ...
```

最终状态标记为 `REWARD_HACK`。

---

## 结果状态

当任意反作弊检查触发时，最终 `Trace.evaluation.status` 会从原本的 `PASSED` 变为 `REWARD_HACK`，同时 `log` 字段包含具体作弊描述。

```json
{
  "evaluation": {
    "status": "REWARD_HACK",
    "log": "Reward hack (ast_scan): Forbidden API torch.matmul\nReward hack (output_caching): Output does not vary across inputs"
  }
}
```

> `REWARD_HACK` 不会替换数值错误状态。如果算子本身已经 `INCORRECT_SHAPE` / `INCORRECT_NUMERICAL` / `RUNTIME_ERROR`，则优先保留这些错误状态。

---

## 配置建议

### 标准评测（推荐）

```json
{
  "enable_anti_hack": ["ast_scan", "kernel_signature"]
}
```

适用于日常开发调试，开销小，能拦截大部分源码级与 kernel 级作弊。

### 严格评测（比赛/打榜）

```json
{
  "enable_anti_hack": [
    "ast_scan",
    "kernel_signature",
    "output_caching",
    "monkey_patch",
    "clock_anomaly"
  ]
}
```

适用于需要高度可信结果的评测场景。`output_caching` 与 `clock_anomaly` 会带来少量额外执行开销。

### 关闭反作弊

```json
{
  "enable_anti_hack": []
}
```

仅在不信任外部检查或本地调试阶段使用。

---

## 常见问题

### Q1: 为什么我的合法 `torch_npu.npu_rotary_mul` 被 AST 扫描拦截？

请确保使用的是最新版 `anti_hack.py`。`torch_npu.npu_*` 系列自定义算子已被加入白名单。

### Q2: `kernel_signature` 为什么会跳过我的纯 PyTorch 实现？

L1 kernel 签名校验只在源码中存在 `@triton.jit` 或 `__global__` kernel 定义时才生效。纯 PyTorch 实现没有这类声明，因此自动跳过。

### Q3: `monkey_patch` 为什么没有生效？

检查 `definition.json` 中 `reference_hash` 是否已正确设置。如果为空，检查会被跳过。

### Q4: 可以同时触发多个反作弊检查吗？

可以。所有检查独立运行，log 中会按顺序列出所有检测到的作弊项。

---

## 扩展新的反作弊检查

如需新增检查：

1. 在 `src/npu_kernelbench/evaluator/anti_hack.py` 中实现检测函数，返回 `RewardHackResult` 或 `list[str]`。
2. 在 `src/npu_kernelbench/evaluator/evaluator.py` 的 `Evaluator.evaluate()` 中调用。
3. 在 `src/npu_kernelbench/runner/eval.py` 的独立复现脚本中同步调用。
4. 将新检查名称加入 `BenchmarkConfig.enable_anti_hack` 文档与 CLI 选项。
5. 在本文档中补充说明。
