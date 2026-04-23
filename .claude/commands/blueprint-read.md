# /blueprint-read

读取 BLUEPRINT.md 的完整内容到上下文，用于任务开始前的代码定位。

## 执行步骤

1. 读取 BLUEPRINT.md 全文
2. 输出"目录树与模块职责"和"关键函数/类位置索引"章节
3. 如果用户提供了关键词（如 `/blueprint-read subgraph`），额外高亮匹配的条目

## 使用时机

- 任务开始前必须执行（CLAUDE.md 强制要求）
- 找代码位置时优先用本命令，而不是 grep
