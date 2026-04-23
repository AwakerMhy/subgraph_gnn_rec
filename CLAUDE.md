# Temporal Social Link Prediction via 2-Hop Subgraph GNN — CLAUDE.md

> 创建时间：2026-04-08 15:30
>
> **本文件是 Claude Code 在本项目中的行为规范，每次会话开始时自动加载。**

---

## 项目概述

基于二度邻居子图的 GNN 链接预测研究项目。面向同质有向社交图，预测候选节点在未来一段时间内向推荐节点建立有向边的概率。边的时间戳仅用于数据集划分，不作为模型输入特征。

---

## 项目结构

```
project/
├── CLAUDE.md                     # 本文件（项目规范）
├── BLUEPRINT.md                  # 目录结构 + 模块职责 + 关键函数索引（每次任务前必读）
├── DECISIONS.md                  # 架构决策记录（ADR）
├── MISTAKES.md                   # 错误与教训库
├── PROGRESS.md                   # 当前任务进度
├── TODO.md                       # 中长期任务与研究计划
├── CHECKPOINT.md                 # 会话中断恢复点
├── META_REFLECTION.md            # 周期性元反思
├── PLAN.md                       # 完整项目计划（只读参考）
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── dataset/
│   └── model/
├── data/
│   ├── raw/                      # 原始下载（不进 git）
│   ├── processed/                # 预处理后标准格式
│   └── synthetic/                # 合成数据集输出
├── src/
│   ├── dataset/
│   ├── graph/
│   ├── model/
│   ├── baseline/
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
├── experiments/
├── notebooks/
├── results/
│   ├── logs/
│   ├── checkpoints/
│   └── tables/
├── tests/
└── docs/
    ├── progress.md               # 历史变更日志
    └── reproducibility.md        # 可复现性基建规范
```

> 详细模块职责、关键函数位置、数据流图见 `BLUEPRINT.md`。

---

## 项目初始化清单 [BLOCKING]

新项目首次启用时，CC **必须**按顺序完成 A/B/C 三步。**未完成视为初始化未结束**。

### A. 配套 md 文件

| 文件 | 用途 | 何时读 | 何时写 |
|---|---|---|---|
| `BLUEPRINT.md` | 目录树 + 模块职责 + 关键函数 `file:line` 索引 | **每次任务开始前必读** | 代码结构变化后立即更新 |
| `DECISIONS.md` | 架构决策记录（ADR） | 做新决策前必读 | 做出非平凡技术决策后 |
| `MISTAKES.md` | 错误与教训库 | 任务开始前扫读 | 每次犯错后立即追加 |
| `PROGRESS.md` | 当前任务子步骤进度 | 任务进行中 | 每个子步骤完成后 |
| `TODO.md` | 中长期任务与研究计划 | 规划新任务时 | 计划变更时 |
| `docs/progress.md` | 历史变更日志 | 回溯历史时 | 每次完成功能/修 Bug 后 |
| `CHECKPOINT.md` | 会话中断恢复点 | 会话恢复时 | 收到结束信号时 |
| `META_REFLECTION.md` | 周期性元反思 | 里程碑节点 | 每 ~20 条 mistake 触发 |
| `docs/reproducibility.md` | 可复现性基建规范 | 训练/数据 pipeline 任务前 | 规范升级时 |

### B. Hooks（见 `.claude/settings.json`）

已配置五类 hook，脚本在 `.claude/hooks/` 目录下：
- `UserPromptSubmit`：注入 BLUEPRINT/MISTAKES/DECISIONS/PROGRESS 上下文
- `PreToolUse`：拦截危险命令、大文件 cat、后台训练
- `PostToolUse`：检查 md 日期标注、提醒同步 BLUEPRINT
- `Stop`：检查 PROGRESS 更新、提示追加 MISTAKES
- `SessionEnd`：dump 状态到 CHECKPOINT.md

### C. Slash Commands（见 `.claude/commands/`）

`/blueprint-update`、`/blueprint-read`、`/decision`、`/mistake`、`/progress-init`、`/progress-tick`、`/checkpoint`、`/simplify-pass`、`/meta-reflect`

---

## 任务开始前的强制流程 [BLOCKING]

任何实质性任务开始前，**必须**按顺序执行：

1. **读 `BLUEPRINT.md`**：定位需修改的模块，避免盲目 grep
2. **扫 `MISTAKES.md`**：检查"检测信号"是否匹配当前任务
3. **查 `DECISIONS.md`**：确认当前任务不会推翻已有决策
4. **初始化 `PROGRESS.md`**（或 `/progress-init`）

**违规后果**：跳过任一步骤的代码改动**视为未完成**，必须回退重走。

**失败检测信号**：
- 没读 `BLUEPRINT.md` 就用 `Grep` 找代码位置
- 声称"我之前没遇到过这个问题"但没看 `MISTAKES.md`
- 写代码前 `PROGRESS.md` 没有对应任务条目
- 推翻一个看起来"奇怪"的设计但没查 `DECISIONS.md`

---

## 范围控制：反过度工程 [BLOCKING]

CC 最常见的失败模式是"做多了"。以下行为**严格禁止**：

- 修一个 bug 不需要顺手清理周围代码
- 不为不存在的需求设计抽象（等第四次再抽函数）
- 不给内部代码加防御性校验（只在系统边界做校验）
- 不加 try/except 兜底不可能发生的异常
- 不写"以防万一"的回退逻辑、feature flag、向后兼容 shim
- 不给没改的代码加注释、docstring、类型注解
- 不在 bug fix 中夹带重构

**失败检测信号**：diff 包含用户没提到的文件；PR 描述里出现"顺便"、"另外"、"我注意到"。

---

## 何时必须停下来问用户 [BLOCKING]

- 模板占位符不知道填什么
- 同一目标有 ≥2 种合理实现路径且影响后续架构
- 即将做不可逆操作（删数据、删 ckpt、reset --hard）
- 修改超出当前任务范围的代码
- 需要引入新的依赖库

---

## 项目蓝图维护 [BLOCKING]

`BLUEPRINT.md` 是项目"地图"，核心目的是显著减少 grep/read 的 token 开销。

**必须包含**：目录树（含职责）、模块依赖关系、关键函数位置索引（`file:line`）、核心数据流、关键数据结构 schema、已知扩展点。

**维护规则**：代码结构改动后**立即**更新，与代码改动属于同一原子提交。

---

## 决策记录 DECISIONS.md [BLOCKING]

**记录格式**（ADR 简化版）：
```
## [YYYY-MM-DD] 决策标题
- **背景**：当时面临什么问题
- **备选方案**：考虑过哪几种，每种利弊
- **决定**：选了哪个
- **原因**：为什么
- **后果**：这个选择带来的限制和后续影响
- **状态**：active / superseded by [...]
```

---

## 错误与教训记录 [BLOCKING]

**记录格式**：
```
## [YYYY-MM-DD] 简短标题
- **类别**：思路错误 / 代码错误 / 工具误用 / 需求误解
- **现象**：发生了什么
- **根因**：为什么会犯
- **教训**：下次应该怎么做
- **检测信号**：什么情况触发警觉
- **复发次数**：1
```

每 ~20 条触发 `/meta-reflect`。

---

## 开发规范

### 环境配置

- **Conda 环境**：`gnn` （`conda create -n gnn python=3.10`）
- **依赖安装**：统一在 `gnn` conda 环境中，不得使用 base 或系统 Python
- **torch 安装**：使用这个文件:"C:\Users\12143\Desktop\pythonProject\torch-2.11.0+cu128-cp310-cp310-win_amd64.whl"

### 技术栈

```
torch >= 2.0
dgl >= 2.1          # GNN 框架（非 PyG）
numpy, pandas, scipy, scikit-learn
pyyaml, tqdm, matplotlib, jupyter
```

### DGL API 约定

- 子图：`dgl.graph((src, dst))`，特征挂载为 `g.ndata['feat']`
- 有向图消息传递：入边 `g.in_edges`，出边 `dgl.reverse(g)`
- Readout：`dgl.mean_nodes(g, 'feat')`
- 批量子图：`dgl.batch([g1, g2, ...])`

### 代码风格

- Python 3.10+，使用类型注解
- 模块导入用绝对路径
- Tensor 操作注意 dtype 一致性

### 核心概念

- **截断图** $\mathcal{G}_{t_q}$：仅包含 $t < t_q$ 的历史边（严格防止时间泄露）
- **子图** $\mathcal{S}_{uv}$：$u$ 和 $v$ 各自二度邻居并集所诱导的有向子图
- **DRNL 标记**：每个节点按到 $u$、$v$ 的最短路径距离赋予离散标签，嵌入为向量后与节点特征拼接
- **度特征占位**：真实数据集无原生属性时，用 [in_degree, out_degree, total_degree]（3维）占位

---

## 测试规范 [MUST]

- **数据 pipeline**：必须有形状/范围/dtype 断言
- **关键算法函数**：`tests/` 下有 unit test，commit 前跑
- **TDD 微规则**：修 bug 时先写复现 bug 的测试，再修代码
- **集成测试**：禁止用 mock 替代真实文件 I/O（外部服务除外）
- **核心模块行覆盖** ≥ 80%

---

## 可复现性基建 [BLOCKING]

详细规范见 **`docs/reproducibility.md`**。五大支柱：

1. **依赖锁定**：`environment.yml` + `requirements.txt` 双锁
2. **随机种子**：统一 `src/utils/seed.py` 中的 `set_seed()` 入口，种子写入 config
3. **实验目录**：`results/logs/<timestamp>_<name>/` 自包含（config snapshot / git hash / env / seed / metrics.json / train.json / ckpt）
4. **数据版本**：`data/raw/` 只读带哈希校验，`data/processed/` 由脚本生成带 META
5. **配置外置**：超参全走 `configs/*.yaml` + 命令行 override，**禁止硬编码**

---

## 不可逆操作护栏 [MUST]

以下操作必须**用户显式确认**后执行：
- 删除/覆盖 checkpoint、训练结果、数据文件
- `rm -rf`，`git reset --hard`，`git push --force`，删除分支
- 修改 `.gitignore`、CI 配置、依赖锁文件
- 清空缓存目录

---

## 断点续训 [MUST]

所有训练脚本必须支持：
- `--resume <ckpt_path>`：恢复优化器、scheduler、step、随机数生成器
- 每 epoch 保存 ckpt，滚动保留最近 K 个
- 后台运行：`nohup python train.py ... > log.txt 2>&1`（禁止 `&`）

---

## ML 研究规范 [MUST]

### 训练与推理日志

**训练脚本**实时打印：epoch/总 epoch、batch 进度、loss、val loss、val AUC、学习率、耗时。batch 级别至少每 20% 输出一次。

**评测脚本**实时打印：已处理样本数/总数、当前指标、预计剩余时间。

**日志文件**：训练时同步写入 `<run_dir>/logs/train.json`（每 epoch 追加）。

### 子图缓存

子图提取是瓶颈，必须优先离线预计算并缓存：
- 缓存路径：`data/processed/<dataset>/subgraphs/`
- 格式：`dgl.save_graphs` 存 `.bin` 文件
- 缓存键 = 数据集名 + 截断时刻 + max_hop + max_neighbors_per_node

### 时间泄露防护

**所有子图提取调用必须传入 `cutoff_time`，函数内部断言过滤**。违反此规则的代码视为严重错误，立即停止并修复。

---

## 工作流规范

### 任务进度管理 [MUST]

`PROGRESS.md` 记录格式：
```
## [任务名称]
- 状态：进行中 / 已完成
- 开始时间：YYYY-MM-DD HH:MM
- 当前进度：第 N 步 / 共 M 步
- [x] 步骤1
- [ ] 步骤2（← 当前）
- 已变更文件：file1.py, file2.py
- 下一步行动：具体描述
```

### 变更日志

完成功能或修 Bug 后，在 `docs/progress.md` 末尾追加：
```
[YYYY-MM-DD HH:MM] [任务描述] [变更文件] [状态]
```

### 文档日期标注 [MUST]

新建或更新 `.md` 文档时：
- **新建**：标题下方 `> 创建时间：YYYY-MM-DD HH:MM`
- **更新**：顶部或末尾 `> 最后更新：YYYY-MM-DD HH:MM`

### 代码简化巡检

每完成 ~5 个任务或一个里程碑，触发 `/simplify-pass`。

### 会话存档

收到结束信号（"好了"、"先这样"、"暂停"）→ 主动 `/checkpoint`。

---

## 安全与机密 [MUST]

- `.env`、`*.key`、`credentials*.json` 必须在 `.gitignore`
- API key 走环境变量加载，禁止硬编码
- 禁止在 md 文件中粘贴 token、密码、API key

---

## 交互与效率规范

- **简洁回复**：仅展示变更部分，不重复输出完整文件，结尾不做总结
- **优先用蓝图定位**：先查 `BLUEPRINT.md`，没有再 grep
- **大文件处理**：用 `grep`/`head`/`tail` 定位，禁止 `cat` 整个大文件
- **命令输出限制**：超过 50 行时追加 `| tail -30` 或针对性 `grep`
- **禁止重复读文件**：同一会话已读过的文件不重复读
