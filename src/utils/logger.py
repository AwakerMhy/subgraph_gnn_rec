"""src/utils/logger.py — 结构化训练日志"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


class TrainLogger:
    """将训练指标写入 <run_dir>/logs/train.json，每 epoch 追加一行。"""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.log_path = self.run_dir / "logs" / "train.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, metrics: dict) -> None:
        """追加一条 epoch 记录。metrics 应包含 epoch、loss 等字段。"""
        metrics["_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def save_final_metrics(self, metrics: dict) -> None:
        """写入最终测试集指标到 <run_dir>/metrics.json。"""
        out = self.run_dir / "metrics.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def save_config(self, cfg_dict: dict) -> None:
        """保存 config 快照到 <run_dir>/config.json。"""
        out = self.run_dir / "config.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
