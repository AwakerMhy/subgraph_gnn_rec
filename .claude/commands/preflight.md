# /preflight

启动实验前的五项强制检查。任意一项失败则停止，修复后重跑。

## 执行步骤

### 1. Smoke test

用最小数据集（或 `rounds=1`）执行一轮：
- 找到本次实验的入口脚本（通常是 `src/loop.py` 或同类脚本）
- 临时设置 `rounds=1`，用规模最小的数据集运行
- 确认脚本从头跑到尾，无报错、无 NaN loss

失败条件：脚本崩溃、报错退出、出现 `NaN`/`inf`。

### 2. out_dir 唯一性

读取实验 config 中的 `out_dir` 字段，执行：
```powershell
Test-Path "<out_dir>"
```
- 若返回 `True`：**停止**，提示用户更新时间戳或 variant 标签
- 若返回 `False`：通过

### 3. 编码环境

```powershell
$env:PYTHONIOENCODING
```
- 若输出不是 `utf-8`：执行 `$env:PYTHONIOENCODING = "utf-8"` 并告知用户
- 确认后通过

### 4. GPU 可用性（仅 GNN 模型需要）

```powershell
$env:PYTHONIOENCODING = "utf-8"; & "C:\Users\12143\miniconda3\envs\gnn\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```
- 若返回 `False`：警告用户，询问是否继续 CPU 运行
- 若返回 `True`：通过

### 5. 首轮指标合理性

smoke test 完成后检查第 1 轮输出中的指标：

| 指标 | 异常条件 |
|------|----------|
| `mrr@k` | 等于 `1.0` 或等于 `0.0` |
| `coverage` | 等于 `0.0` 或等于 `1.0` |
| `auc` | 等于 `1.0` 或等于 `0.5`（随机基线除外）|

任意指标触发异常条件 → **立即停止**，输出具体数值，提示用户排查。

## 通过标准

五项全部通过后，输出：

```
[preflight] ✓ 所有检查通过，可以启动全量实验。
```

## 失败处理

- 每项失败单独报告，列出失败原因和修复建议
- 不跳过任何失败项，不在用户确认前继续
