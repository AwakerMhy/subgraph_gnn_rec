"""scripts/run_ablation_grid.py — 消融实验网格调度器。

用法：
    # 中心点 + 单因子翻转（约 12 runs）
    python -m scripts.run_ablation_grid --mode center_plus_flips

    # 完整 216 runs 笛卡尔积（慎用）
    python -m scripts.run_ablation_grid --mode full

结果写入 results/ablation/summary.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from itertools import product
from pathlib import Path

import pandas as pd
import yaml


# ── 消融维度定义 ──────────────────────────────────────────────────────────────

CENTER = {
    "feedback.p_pos": 0.8,
    "feedback.p_neg": 0.02,
    "feedback.cooldown_mode": "decay",
    "recall.method": "mixture",
    "user_selector.strategy": "composite",
    "replay.capacity": 200,
    "model.type": "gnn",
    "model.num_layers": 2,
}

ABLATION_DIMS = {
    "feedback.p_pos+p_neg": [(0.8, 0.02), (1.0, 0.0), (0.5, 0.05)],
    "recall.method":        ["adamic_adar", "ppr", "mixture"],
    "user_selector.strategy": ["uniform", "composite"],
    "feedback.cooldown_mode": ["hard", "decay"],
    "replay.capacity":       [0, 200],
    "model.type":            ["mlp", "gnn"],
}


def set_nested(cfg: dict, dotpath: str, value) -> None:
    keys = dotpath.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_base_config(base_cfg_path: str) -> dict:
    with open(base_cfg_path) as f:
        return yaml.safe_load(f)


def apply_center(cfg: dict) -> None:
    for key, val in CENTER.items():
        if key == "feedback.p_pos+p_neg":
            continue
        set_nested(cfg, key, val)
    set_nested(cfg, "feedback.p_pos", CENTER["feedback.p_pos"])
    set_nested(cfg, "feedback.p_neg", CENTER["feedback.p_neg"])


def run_single(cfg: dict, run_name: str, base_out: Path) -> dict:
    from src.online.loop import run_online_simulation  # noqa: PLC0415

    cfg = copy.deepcopy(cfg)
    cfg["runtime"]["out_dir"] = str(base_out / run_name)
    cfg["runtime"]["log_every"] = 9999  # 静默
    t0 = time.time()
    df = run_online_simulation(cfg)
    elapsed = time.time() - t0
    last = df.iloc[-1]
    return {
        "run_name": run_name,
        "coverage_delta": float(df["coverage"].iloc[-1] - df["coverage"].iloc[0]),
        "coverage_final": float(df["coverage"].iloc[-1]),
        "avg_precision_k": float(df["precision_k"].mean()),
        "total_accepted": int(df["n_accepted"].sum()),
        "elapsed_s": round(elapsed, 1),
        **{k: v for k, v in cfg.get("feedback", {}).items()
           if k in ("p_pos", "p_neg", "cooldown_mode")},
        "recall_method": cfg.get("recall", {}).get("method", ""),
        "user_strategy": cfg.get("user_selector", {}).get("strategy", ""),
        "replay_capacity": cfg.get("replay", {}).get("capacity", 0),
        "model_type": cfg.get("model", {}).get("type", "gnn"),
    }


def build_center_plus_flips(base_cfg: dict) -> list[tuple[str, dict]]:
    """中心点 + 每个维度单因子翻转。"""
    runs = []

    # center run
    c = copy.deepcopy(base_cfg)
    apply_center(c)
    runs.append(("center", c))

    # single-factor flips
    for dim, values in ABLATION_DIMS.items():
        for val in values:
            c2 = copy.deepcopy(base_cfg)
            apply_center(c2)
            if dim == "feedback.p_pos+p_neg":
                set_nested(c2, "feedback.p_pos", val[0])
                set_nested(c2, "feedback.p_neg", val[1])
                run_id = f"ppos{val[0]}_pneg{val[1]}"
            else:
                set_nested(c2, dim, val)
                run_id = f"{dim.split('.')[-1]}_{val}"
            # 跳过中心点自身
            if run_id == "center":
                continue
            runs.append((run_id, c2))

    # 去重
    seen = set()
    deduped = []
    for name, cfg in runs:
        if name not in seen:
            seen.add(name)
            deduped.append((name, cfg))
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="center_plus_flips",
                        choices=["center_plus_flips", "full"])
    parser.add_argument("--base_config",
                        default="configs/online/college_msg_full.yaml")
    parser.add_argument("--out_dir", default="results/ablation")
    args = parser.parse_args()

    base_cfg = load_base_config(args.base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "center_plus_flips":
        run_list = build_center_plus_flips(base_cfg)
    else:
        raise NotImplementedError("full grid not yet implemented — start with center_plus_flips")

    print(f"Running {len(run_list)} configurations ...")
    results = []
    for i, (name, cfg) in enumerate(run_list):
        print(f"[{i+1}/{len(run_list)}] {name}")
        try:
            row = run_single(cfg, name, out_dir)
            results.append(row)
            print(f"  coverage_delta={row['coverage_delta']:.4f}  "
                  f"avg_prec={row['avg_precision_k']:.4f}  "
                  f"({row['elapsed_s']}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"run_name": name, "error": str(e)})

    summary_df = pd.DataFrame(results)
    out_path = out_dir / "summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
