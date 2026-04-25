#!/usr/bin/env python3
"""
PreToolUse hook: 拦截危险操作和违规模式。
- 拦截 & 后台训练
- 拦截危险命令（rm -rf / git reset --hard / git push --force / drop table）
"""
import sys
import json
import re
import os

DANGEROUS_PATTERNS = [
    (r'\brm\s+-rf\b', "rm -rf 是危险操作，请用户显式确认后再执行"),
    (r'\bgit\s+reset\s+--hard\b', "git reset --hard 是不可逆操作，请用户显式确认"),
    (r'\bgit\s+push\s+--force\b', "git push --force 可能覆盖远程历史，请用户显式确认"),
    (r'\bgit\s+push\s+-f\b', "git push -f 可能覆盖远程历史，请用户显式确认"),
    (r'\bDROP\s+TABLE\b', "DROP TABLE 是不可逆操作，请用户显式确认"),
    (r'\bTRUNCATE\s+TABLE\b', "TRUNCATE TABLE 是不可逆操作，请用户显式确认"),
    (r'\bgit\s+branch\s+-D\b', "强制删除分支是不可逆操作，请用户显式确认"),
    (r'\bgit\s+clean\s+-f\b', "git clean -f 会删除未跟踪文件，请用户显式确认"),
]

BG_PATTERN = re.compile(r'python\s+.*train.*&\s*$')

def check_bash_command(command: str) -> str | None:
    """返回 block 原因，None 表示放行"""
    # 拦截危险命令
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"[PreToolUse] 拦截：{reason}\n命令：{command}"

    # 拦截 & 后台训练
    if BG_PATTERN.search(command):
        return (
            "[PreToolUse] 拦截：禁止用 & 后台运行训练脚本。\n"
            f"请改用：nohup python train.py ... > log.txt 2>&1\n命令：{command}"
        )

    return None

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    block_reason = None

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        block_reason = check_bash_command(command)

    if block_reason:
        output = {
            "decision": "block",
            "reason": block_reason
        }
        print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)

    # 放行
    sys.exit(0)

if __name__ == "__main__":
    main()
