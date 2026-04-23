# /blueprint-update

读取最近的 git diff（或当前已变更的文件列表），分析代码结构变化，生成 BLUEPRINT.md 的更新建议。

## 执行步骤

1. 运行 `git diff HEAD --name-only` 查看变更文件
2. 对变更的 `.py` 文件，读取其中的类和函数定义（重点：新增/删除/重命名）
3. 对照现有 BLUEPRINT.md 的"关键函数/类位置索引"和"目录树"章节
4. 生成具体的更新建议（精确到 file:line）
5. 询问用户确认后，更新 BLUEPRINT.md，并在末尾追加 `> 最后更新：YYYY-MM-DD HH:MM`

## 注意

- 只更新有实质变化的条目，不做无谓的重写
- 如果没有 git 历史，直接扫描 src/ 下所有 .py 文件的类/函数定义
