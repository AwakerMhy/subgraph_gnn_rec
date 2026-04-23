"""scripts/run_online_sim.py — 在线学习仿真 CLI 入口。

用法：
    PYTHONPATH=. python scripts/run_online_sim.py --config configs/online/default.yaml
    PYTHONPATH=. python scripts/run_online_sim.py --config configs/online/sbm_smoke.yaml --dry_run
"""
from __future__ import annotations

import argparse
import copy

import yaml

from src.online.loop import run_online_simulation


def _deep_set(d: dict, key: str, value: object) -> None:
    """简单 dot-path 覆盖，如 trainer.lr → d['trainer']['lr'] = value。"""
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="在线社交推荐仿真")
    parser.add_argument("--config", required=True)
    # 常用覆盖项
    parser.add_argument("--dataset",           default=None)
    parser.add_argument("--total_rounds",      type=int, default=None)
    parser.add_argument("--top_k_recall",      type=int, default=None)
    parser.add_argument("--top_k_recommend",   type=int, default=None)
    parser.add_argument("--p_accept",          type=float, default=None)
    parser.add_argument("--cooldown_rounds",   type=int, default=None)
    parser.add_argument("--init_edge_ratio",   type=float, default=None)
    parser.add_argument("--user_sample_ratio", type=float, default=None)
    parser.add_argument("--recall_method",     default=None)
    parser.add_argument("--lr",                type=float, default=None)
    parser.add_argument("--scheduler_strategy",default=None)
    parser.add_argument("--replay_capacity",   type=int, default=None)
    parser.add_argument("--seed",              type=int, default=None)
    parser.add_argument("--device",            default=None)
    parser.add_argument("--out_dir",           default=None)
    parser.add_argument("--log_every",         type=int, default=None)
    parser.add_argument("--dry_run",           action="store_true",
                        help="只跑 1 轮烟测，验证形状无误")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "dataset.type":              args.dataset,
        "total_rounds":              args.total_rounds,
        "recall.top_k_recall":       args.top_k_recall,
        "recommend.top_k":           args.top_k_recommend,
        "feedback.p_accept":         args.p_accept,
        "feedback.cooldown_rounds":  args.cooldown_rounds,
        "init_edge_ratio":           args.init_edge_ratio,
        "user_sample_ratio":         args.user_sample_ratio,
        "recall.method":             args.recall_method,
        "trainer.lr":                args.lr,
        "trainer.scheduler.strategy": args.scheduler_strategy,
        "replay.capacity":           args.replay_capacity,
        "runtime.seed":              args.seed,
        "runtime.device":            args.device,
        "runtime.out_dir":           args.out_dir,
        "runtime.log_every":         args.log_every,
    }
    for k, v in overrides.items():
        if v is not None:
            _deep_set(cfg, k, v)

    if args.dry_run:
        cfg["total_rounds"] = 1
        cfg.setdefault("trainer", {})["min_batch_size"] = 1
        print("[dry_run] 仅跑 1 轮")

    run_online_simulation(cfg)


if __name__ == "__main__":
    main()
