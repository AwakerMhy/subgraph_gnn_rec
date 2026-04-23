# CLAUDE.md 项目模板

> **本文件的用途**：这是用于初始化新项目 `CLAUDE.md` 的**框架与规则集**。当 Claude Code 在新项目中读取本文件时，必须将其作为生成该项目 `CLAUDE.md` 的基线模板，并完成"项目初始化清单"中的全部动作。
>
> **生成时的处理方式**：
> 1. 将所有 `[占位符]` 替换为该项目的真实内容。未知信息**必须主动询问用户**，禁止臆测填写。
> 2. 删除不适用的章节（如非 ML 项目可删除"ML 研究规范"一节）
> 3. **不可删除** 标注 `[MUST]` 或 `[BLOCKING]` 的规范
> 4. 生成完成后，按"项目初始化清单"创建配套的 md 文件骨架、`.claude/settings.json` 中的 Hooks、以及 `.claude/commands/` 下的 slash commands

---

## 项目初始化清单 [BLOCKING]

新项目首次启用时，CC **必须**按顺序完成 A/B/C 三步，缺一不可。

### A. 创建配套 md 文件骨架

| 文件 | 用途 | 何时读 | 何时写 |
|---|---|---|---|
| `CLAUDE.md` | 项目规范与上下文（本模板生成） | 每次会话开始 | 规则变更时 |
| `BLUEPRINT.md` | 项目蓝图：目录结构 + 模块职责 + 关键函数 `file:line` 索引 | **每次任务开始前必读** | 代码结构变化后立即更新 |
| `DECISIONS.md` | 架构决策记录（ADR）：每条决策的背景、备选、原因、后果 | 做新决策前必读，避免推翻已定方案 | 做出非平凡技术决策后立即追加 |
| `MISTAKES.md` | 错误与教训库 | 任务开始前扫读 | 每次犯错后立即追加 |
| `PROGRESS.md` | 当前任务的子步骤进度 | 任务进行中持续读写 | 每个子步骤完成后 |
| `TODO.md` | 中长期任务与研究计划 | 规划新任务时 | 计划变更时 |
| `docs/progress.md` | 历史变更日志 | 回溯历史时 | 每次完成功能/修 Bug 后 |
| `CHECKPOINT.md` | 会话中断时的恢复点 | 会话恢复时 | 收到结束信号 / 遇到 rate limit |
| `META_REFLECTION.md` | 周期性元反思：从 MISTAKES.md 中找模式 | 里程碑节点 | 每 ~20 条 mistake 触发一次 |
| `docs/reproducibility.md` | 可复现性基建外挂规范（依赖/种子/实验目录/数据版本/config） | 训练、数据 pipeline、依赖变更任务前 | 规范升级时 |

### B. 创建 Hooks（强制执行层）

光靠 CC 自觉读规则不够，必须配置 `.claude/settings.json` 让 harness 强制检查。**未配置 Hooks 视为初始化未完成**。

需创建的 hooks（脚本可用 Python 或 bash，CC 应根据用户操作系统选择）：

1. **`UserPromptSubmit`**（每次用户发消息时注入上下文）
   - 注入 `BLUEPRINT.md` 的目录树和函数索引
   - 注入 `MISTAKES.md` 中最近 10 条 + 与提示词关键词匹配的条目
   - 注入 `DECISIONS.md` 标题列表
   - 注入 `PROGRESS.md` 当前任务状态
   - **目的**：消除"CC 忘记读元文件"的可能

2. **`PreToolUse`**（拦截层）
   - 拦 `Bash`：`cat <file>` 且文件 > 200 行 → block，提示用 `grep`/`head`/`tail`
   - 拦 `Bash`：`&` 后台运行训练脚本 → block，提示用 `nohup ... > log.txt 2>&1`
   - 拦 `Bash`：危险命令（`rm -rf`、`git reset --hard`、`git push --force`、`drop table`、删 ckpt/数据）→ block 并要求用户显式确认
   - 拦 `Edit`/`Write`：代码文件改动累积 ≥3 次而 `BLUEPRINT.md` 未同步 → warn

3. **`PostToolUse`**（补救层）
   - 写 `.md` 文件后检查日期标注，缺失则自动补
   - 写代码文件后向下一轮注入提醒："评估是否需同步 `BLUEPRINT.md`"

4. **`Stop`**（结束检查）
   - 本轮有代码改动但 `PROGRESS.md` 未更新 → 下一轮强制提示
   - 本轮触发了错误或用户使用纠正性措辞（"不对"/"错了"/"应该"）→ 提示追加 `MISTAKES.md`

5. **`SessionEnd`**
   - 自动 dump 状态到 `CHECKPOINT.md`：未完成任务、变更文件、下一步行动

CC 应生成 `.claude/settings.json` 草稿和 `.claude/hooks/` 下脚本，**完成后向用户展示并请求确认**，禁止默默写入。

### C. 创建 Slash Commands

在 `.claude/commands/` 下创建：

| 命令 | 功能 |
|---|---|
| `/blueprint-update` | 读 git diff，生成 `BLUEPRINT.md` 更新建议 |
| `/blueprint-read` | 读取 `BLUEPRINT.md` 索引到上下文 |
| `/decision <标题>` | 按 ADR 模板向 `DECISIONS.md` 追加新决策 |
| `/mistake <标题>` | 按格式追加一条 mistake 模板 |
| `/progress-init <任务名>` | 初始化新任务区块 |
| `/progress-tick` | 推进当前任务下一步 |
| `/checkpoint` | 写入 `CHECKPOINT.md` |
| `/simplify-pass` | 读最近 N 个 commit 的 diff，找重复/死代码/过度抽象 |
| `/meta-reflect` | 扫 `MISTAKES.md` 找模式，写入 `META_REFLECTION.md` |

---

## 任务开始前的强制流程 [BLOCKING]

任何实质性任务开始前，**必须**按顺序执行：

1. **读 `BLUEPRINT.md`**（或 `/blueprint-read`）：定位需修改的模块，避免盲目 grep
2. **扫 `MISTAKES.md`**：检查"检测信号"是否匹配当前任务
3. **查 `DECISIONS.md`**：确认当前任务不会推翻已有决策；若必须推翻，先与用户讨论更新决策
4. **初始化 `PROGRESS.md`**（或 `/progress-init`）

**违规后果**：跳过任一步骤的代码改动**视为未完成**，必须回退重走。

**失败检测信号**（如果你正在做以下事情，立即停手）：
- 没读 `BLUEPRINT.md` 就用 `Grep` 找代码位置
- 声称"我之前没遇到过这个问题"但没看 `MISTAKES.md`
- 写代码前 `PROGRESS.md` 没有对应任务条目
- 推翻一个看起来"奇怪"的设计但没查 `DECISIONS.md`

---

## 范围控制：反过度工程 [BLOCKING]

CC 最常见的失败模式是"做多了"。以下行为**严格禁止**：

- **不要做用户没要求的事**。修一个 bug 不需要顺手清理周围代码；加一个功能不需要顺手加配置项
- **不要为不存在的需求设计抽象**。三段相似代码不算重复，不要急着抽函数。等到第四次再考虑
- **不要给内部代码加防御性校验**。信任内部调用方，只在系统边界（用户输入、外部 API、文件 I/O）做校验
- **不要加 try/except 兜底不可能发生的异常**。让它崩，崩了再说
- **不要写"以防万一"的回退逻辑、feature flag、向后兼容 shim**，除非用户明确要求
- **不要给没改的代码加注释、docstring、类型注解**
- **不要在 bug fix 中夹带重构**

**失败检测信号**：
- 你的 diff 包含用户没提到的文件
- 你正在写"未来可能需要"的代码
- 你正在为一段只调用一次的逻辑创建辅助函数 / 工具类
- 你的 PR 描述里出现"顺便"、"另外"、"我注意到"

违反此规则的改动应主动回退到最小必要集。

---

## 何时必须停下来问用户 [BLOCKING]

CC 倾向于猜测前进。以下情形**必须停下来问，禁止猜测**：

- 模板占位符不知道填什么（项目名、目录、依赖版本等）
- 同一目标有 ≥2 种合理实现路径，且选择会影响后续架构
- 即将做不可逆操作（删数据、删 ckpt、reset --hard、覆盖结果文件）
- 修改超出当前任务范围的代码
- 用户需求与已有代码、`BLUEPRINT.md`、`DECISIONS.md` 冲突
- 任务描述中存在歧义术语，且不同理解会导致不同实现
- 需要引入新的依赖库

问比猜的成本低得多。

---

## 项目蓝图维护 [BLOCKING]

`BLUEPRINT.md` 是项目的"地图"，**核心目的是显著减少 grep/read 的 token 开销**。蓝图不是文档，是性能优化工具。

**必须包含**：
- 目录树（每个目录/关键文件附一行职责）
- 模块依赖关系（谁调用谁）
- 关键函数/类位置索引：`功能描述 → file_path:line_number`
- 核心数据流：从输入到输出的关键路径
- 关键数据结构 schema（字段名 + 类型 + 含义）
- 已知扩展点

**维护规则**：
- 任何改变代码结构的操作完成后**立即**更新，不允许累积
- 蓝图更新与代码改动属于同一原子提交

**失败检测信号**：
- 刚改完代码却没碰 `BLUEPRINT.md` → 违规
- 用户问"X 功能在哪"，你需要 grep 才能回答 → 蓝图不完整

---

## 决策记录 DECISIONS.md [BLOCKING]

`DECISIONS.md` 是跨会话的长期记忆，防止 CC 反复推翻已经讨论过的方案。**比 MISTAKES.md 更重要**。

**触发记录的时机**：
- 选定一种实现路径而放弃另一种
- 引入新依赖 / 新模块 / 新文件组织方式
- 与用户讨论后达成的非平凡技术约定

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

**使用规则**：
- 做新决策前必须扫读，遇到看似"奇怪"的设计先查这里
- 推翻旧决策必须显式标记 `superseded by`，不允许悄悄改

---

## 错误与教训记录 [BLOCKING]

CC 会反复犯错。所有错误必须沉淀到 `MISTAKES.md`，否则同类错误无限复发。

**触发时机**：
- 用户用纠正性措辞（"不对"、"错了"、"应该"、"为什么"、"重新"）
- 测试 / 运行失败
- 调试中走了明显弯路
- 自己回看发现的判断偏差

**记录格式**：
```
## [YYYY-MM-DD] 简短标题
- **类别**：思路错误 / 代码错误 / 工具误用 / 需求误解
- **现象**：发生了什么、报错信息或用户原话
- **根因**：为什么会犯（追到底层假设）
- **教训**：下次类似情况应该怎么做
- **检测信号**：什么样的代码 / 任务 / 上下文应该触发警觉
- **关联文件**：（可选）
- **复发次数**：1
```

**使用规则**：
- 任务开始前扫读，注意"检测信号"匹配当前任务
- 同类错误第二次发生时 `复发次数 +1`，并反思为何上次教训没起作用——这是元学习
- 每 ~20 条触发 `/meta-reflect` 找模式，写入 `META_REFLECTION.md`，并合并相似条目

---

## 测试规范 [MUST]

文档没有测试就没有可重复性。

- **数据 pipeline**：必须有形状/范围/dtype 断言，输入输出契约用 dataclass / Pydantic 校验
- **关键算法函数**：必须有 `tests/` 下 unit test，commit 前跑
- **TDD 微规则**：修 bug 时**先**写一个能复现 bug 的测试，再修代码——这是防止 MISTAKES.md 同类错误复发的最强机制
- **集成测试**：禁止用 mock 替代真实数据库 / 文件 I/O，除非外部服务无法访问
- **测试覆盖**：核心模块行覆盖 ≥ 80%，非核心可放宽，但禁止 0 测试

---

## 可复现性基建 [BLOCKING]

研究项目的脊柱。详细规范在 **`docs/reproducibility.md`**，CC 在以下时机必须完整读取该文件：项目初始化、涉及训练/数据 pipeline/依赖/config 的任务开始前、引入新随机性来源时。

**五大支柱**（摘要，细则见外挂文件）：
- **依赖锁定**：`environment.yml` + `requirements.lock` 双锁
- **随机种子**：统一 `set_seed()` 入口，种子写 config
- **实验目录**：`runs/<timestamp>_<name>/` 自包含（config / git / env / seed / metrics / logs / ckpt）
- **数据版本**：`data/raw/` 只读带哈希校验，`data/processed/` 由脚本生成带 META
- **配置外置**：超参走 `configs/*.yaml` + 命令行 override，禁止硬编码

**违反任一条视为破坏可复现性**。

---

## 不可逆操作护栏 [MUST]

以下操作必须**用户显式确认**后才能执行（由 `PreToolUse` hook 强制兜底，但 CC 应主动遵守）：

- 删除 / 覆盖 checkpoint、训练结果、数据文件
- `rm -rf`、`rm` 多个文件
- `git reset --hard`、`git clean -f`、`git push --force`、删除分支
- 数据库 `DROP` / `TRUNCATE`
- 修改 `.gitignore`、CI 配置、依赖锁文件
- 清空缓存目录（除非是为了正确重建）

确认时必须告知"将要删除/覆盖的具体内容"和"是否可恢复"。

---

## 缓存与中间产物 [MUST]

昂贵计算（embeddings、预处理数据集、特征提取）必须可缓存：

- 缓存键 = 输入哈希 + 代码版本（git commit）
- `cache/` 目录可随时整体删除并自动重建
- **禁止把缓存当数据源**——缓存丢了脚本必须能重跑出完全相同结果
- `cache/` 必须在 `.gitignore` 中

---

## 断点续训 [MUST]

所有长时间训练脚本必须支持：

- `--resume <ckpt_path>`：从 checkpoint 恢复优化器状态、scheduler、step、随机数生成器
- 每 N steps（或每 epoch）保存 ckpt，旧 ckpt 滚动保留最近 K 个
- 崩溃后能从最近 ckpt 无损继续

与"前台运行"规则配合：长训练用 `nohup python train.py ... > log.txt 2>&1`，崩溃后用 `--resume` 恢复。

---

## 安全与机密 [MUST]

- `.env`、`*.key`、`credentials*.json`、`secrets/` 必须在 `.gitignore`，初始化时建立
- API key 和 token 走环境变量加载，禁止硬编码到 `.py` 或 config
- 写 `MISTAKES.md` / `CHECKPOINT.md` / 任何 md 时禁止粘贴 token、密码、API key
- 提交前用 hook 扫描 diff 中的机密模式（`sk-...`、`AKIA...`、`-----BEGIN PRIVATE KEY-----` 等）

---

## 外部依赖容错 [MUST]

调用外部服务（HuggingFace、API、远程数据源）时：

- 必须设置 `timeout`，禁止裸 `requests.get(url)`
- 必须设置重试上限（如 max_retries=3，指数退避）
- 失败时降级路径明确：报错 / 回退到本地缓存 / 跳过单条数据
- 禁止无限重试导致脚本卡死

---

## 项目概述

[一句话描述项目目标和范围]

---

## 项目结构

```
[粘贴项目目录树]
```

> 详细模块职责、关键函数位置、数据流图见 `BLUEPRINT.md`。

---

## 开发规范

### 环境配置

[说明运行环境要求，例如：]
- Python 版本：3.10+
- GPU/CUDA：使用前 `nvidia-smi` 确认 GPU 架构，安装匹配版本 PyTorch
- 依赖：在 conda 环境中进行，不得使用系统 Python 或 base 环境

### 代码风格

- Python 3.10+，使用类型注解
- 模块导入用绝对路径
- Tensor 操作注意 dtype 一致性，涉及 `torch.erf` 等函数时显式转换

### 核心概念

[项目核心概念说明]

---

## 工作流规范

### 任务进度管理 [MUST]

`PROGRESS.md` 记录格式：
```
## [任务名称]
- 状态：进行中 / 已完成
- 开始时间：YYYY-MM-DD HH:MM
- 当前进度：第 N 步 / 共 M 步
- [x] 步骤1：描述
- [ ] 步骤2：描述（← 当前）
- 已变更文件：file1.py, file2.py
- 下一步行动：具体描述
```

- 每完成一个子步骤立即更新（✅ / 🔄 / ❌）
- 切换任务前先把当前状态写入
- 优先用 `/progress-tick`

### 变更日志

完成功能或修 Bug 后，在 `docs/progress.md` 末尾追加：
```
[YYYY-MM-DD HH:MM] [任务描述] [变更文件] [状态]
```

### 文档日期标注 [MUST]

新建或更新 `.md` 文档时：
- **新建**：标题下方 `> 创建时间：YYYY-MM-DD HH:MM`
- **更新**：顶部或末尾 `> 最后更新：YYYY-MM-DD HH:MM`，保留历史

### 核心文档同步 [MUST]

[列出需要与代码同步的文档]
- 当 `[核心模块]` 变更时，必须同步更新 `[对应文档]`

### 代码简化巡检

每完成 ~5 个任务或一个里程碑，触发 `/simplify-pass`：
- 读最近 N 个 commit 的 diff
- 找重复代码、死代码、过度抽象、未使用的 import / 变量
- 提出简化建议，用户确认后改

### 会话存档

- 收到结束信号（"好了"、"先这样"、"暂停"）→ 主动 `/checkpoint`
- API rate limit 或不可恢复错误时同样保存

### 研究计划

- TODO 与进展：`TODO.md`
- 修改研究方向同步更新

---

## 交互与效率规范

- **简洁回复**：仅展示变更部分，不重复输出完整文件，结尾不做总结
- **优先用蓝图定位**：先查 `BLUEPRINT.md`，没有再 grep
- **大文件处理**：用 `grep`/`head`/`tail` 定位，禁止 `cat` 整个大文件（`PreToolUse` hook 强制）
- **命令输出限制**：超过 50 行时追加 `| tail -30` 或针对性 `grep`
- **禁止重复读文件**：同一会话已读过的文件不重复读

---

## [可选] ML 研究规范

> 适用于机器学习研究项目，非 ML 项目可删除。

### 数据生成

生成数据集后必须验证：
1. 所有 token 存在于词表
2. 答案分布均衡
3. 至少抽样 10 条验证正确性

### 实验管理

- 多优化方向并行实验时，先对单一方向做快速 sanity check 再全量展开
- 每次 run 严格遵守"实验目录约定"（见 可复现性基建）
- 实验结果指标必须写入 `runs/<...>/metrics.json`，禁止只在终端打印

### 训练与推理日志 [MUST]

**训练脚本**必须实时打印：epoch/总 epoch、batch 进度、loss、val loss、val acc、学习率、耗时。batch 级别至少每 20% 输出一次。

**推理/评测脚本**必须实时打印：已处理样本数/总数、当前准确率、预计剩余时间。

**后台运行**：禁止 `&` 后台运行（`PreToolUse` hook 强制）。长训练用 `nohup python train.py ... > log.txt 2>&1`，配合 `--resume` 支持崩溃恢复。

**日志文件**：训练时同步写入 `<run_dir>/logs/train.json`（每 epoch 追加）。
