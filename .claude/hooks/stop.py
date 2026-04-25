#!/usr/bin/env python3
"""
Stop hook: 结束检查。
- 用户使用纠正性措辞 → 提示追加 MISTAKES.md
"""
import sys
import json
import os
from datetime import datetime

CORRECTION_KEYWORDS = ["不对", "错了", "应该", "为什么这样", "重新", "你搞错了", "不是这样"]

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    messages = []

    # 检查用户消息中是否有纠正性措辞
    transcript = data.get("transcript", [])
    last_user_msg = ""
    for msg in reversed(transcript):
        if msg.get("role") == "user":
            last_user_msg = str(msg.get("content", ""))
            break

    if any(kw in last_user_msg for kw in CORRECTION_KEYWORDS):
        messages.append(
            "[Stop] 检测到纠正性措辞，请将本次错误追加到 MISTAKES.md：\n"
            "  - 类别、现象、根因、教训、检测信号"
        )

    if messages:
        output = {"reminders": messages}
        print(json.dumps(output, ensure_ascii=False))

    sys.exit(0)

if __name__ == "__main__":
    main()
