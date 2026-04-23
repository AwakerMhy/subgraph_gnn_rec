# /checkpoint

将当前会话状态 dump 到 CHECKPOINT.md。

## 执行步骤

1. 读取 PROGRESS.md，提取当前未完成任务
2. 读取最近修改的文件列表
3. 更新 CHECKPOINT.md：
   - 存档时间（当前时间）
   - 当前阶段（从 PROGRESS.md 提取）
   - 未完成任务列表
   - 已变更文件列表
   - 下一步行动（从 PROGRESS.md 的"下一步行动"字段提取）

## 使用时机

- 用户说"好了"、"先这样"、"暂停"、"明天继续"
- 遇到 API rate limit 或不可恢复错误
- 任何需要中断会话的时刻

## 注意

- CHECKPOINT.md 中不得包含代码、token、密码
- 只记录状态摘要，不记录具体实现细节（细节在代码里）
