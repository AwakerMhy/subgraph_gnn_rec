#!/usr/bin/env python3
"""
UserPromptSubmit hook: 每次用户发消息时注入关键上下文。
向 CC 注入：BLUEPRINT 目录树、MISTAKES 最近10条、DECISIONS 标题列表、PROGRESS 当前状态。
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_file_lines(path, max_lines=None):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if max_lines:
            lines = lines[:max_lines]
        return "".join(lines).strip()
    except FileNotFoundError:
        return f"[{os.path.basename(path)} 不存在]"

def extract_headings(path, level="##"):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return [l.strip() for l in lines if l.startswith(level + " ")]
    except FileNotFoundError:
        return []

def main():
    context_parts = []

    # BLUEPRINT 前 60 行（目录树部分）
    bp = read_file_lines(os.path.join(BASE, "BLUEPRINT.md"), max_lines=60)
    if bp:
        context_parts.append(f"=== BLUEPRINT (前60行) ===\n{bp}")

    # DECISIONS 标题列表
    decisions = extract_headings(os.path.join(BASE, "DECISIONS.md"))
    if decisions:
        context_parts.append("=== DECISIONS 标题 ===\n" + "\n".join(decisions))

    # MISTAKES 最近 10 条标题
    mistakes = extract_headings(os.path.join(BASE, "MISTAKES.md"))
    recent = mistakes[-10:] if len(mistakes) > 10 else mistakes
    if recent:
        context_parts.append("=== MISTAKES 最近10条 ===\n" + "\n".join(recent))

    # PROGRESS 全文
    prog = read_file_lines(os.path.join(BASE, "PROGRESS.md"), max_lines=40)
    if prog:
        context_parts.append(f"=== PROGRESS ===\n{prog}")

    if context_parts:
        injected = "\n\n".join(context_parts)
        output = {
            "context": injected
        }
        print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
