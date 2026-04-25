#!/usr/bin/env python3
"""
PostToolUse hook: 补救层。
- 写 .md 文件后检查日期标注
"""
import sys
import json
import os
from datetime import datetime

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    warnings = []

    if tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "")
        _, ext = os.path.splitext(file_path)

        if ext == ".md" and not any(
            skip in file_path for skip in ["CHECKPOINT", "META_REFLECTION"]
        ):
            # 检查是否有日期标注
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read(500)
                if "创建时间" not in content and "最后更新" not in content:
                    warnings.append(
                        f"[PostToolUse] 提醒：{os.path.basename(file_path)} 缺少日期标注。"
                        f"请在标题下添加：> 最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    )
            except Exception:
                pass


    if warnings:
        output = {"warnings": warnings}
        print(json.dumps(output, ensure_ascii=False))

    sys.exit(0)

if __name__ == "__main__":
    main()
