"""对比 constant_lr vs cosine_warmup 在所有数据集上的 GNN 指标。

用法:
    python scripts/compare_constant_lr.py --seeds 0
    python scripts/compare_constant_lr.py --seeds 0,1,2 --metric mrr_at_1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = [
    "college_msg", "email_eu", "bitcoin_alpha", "dnc_email", "wiki_vote",
    "slashdot", "sx_mathoverflow", "sx_askubuntu", "sx_superuser", "advogato",
]

GNN_ALGOS = ["gnn", "gnn_h32", "gnn_concat", "gnn_sum"]


def load_rounds_avg(out_dir: Path, metric: str) -> float | None:
    rounds_csv = out_dir / "rounds.csv"
    if not rounds_csv.exists():
        return None
    df = pd.read_csv(rounds_csv)
    if metric not in df.columns:
        return None
    return float(df[metric].mean())


def sweep_name_cosine(dataset: str, seed: int) -> str:
    return f"algo_sweep_{dataset}_v2_s{seed}"


def sweep_name_constlr(dataset: str, seed: int) -> str:
    return f"algo_sweep_{dataset}_v2_constlr_s{seed}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--metric", default="mrr_at_1")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    metric = args.metric

    rows = []
    for ds in DATASETS:
        for algo in GNN_ALGOS:
            tag = f"{ds}_{algo}"
            cosine_vals, const_vals = [], []
            for sd in seeds:
                cosine_dir = Path(f"results/online/{sweep_name_cosine(ds, sd)}/{tag}")
                const_dir  = Path(f"results/online/{sweep_name_constlr(ds, sd)}/{tag}")
                v_cos = load_rounds_avg(cosine_dir, metric)
                v_con = load_rounds_avg(const_dir,  metric)
                if v_cos is not None:
                    cosine_vals.append(v_cos)
                if v_con is not None:
                    const_vals.append(v_con)

            cosine_mean = np.mean(cosine_vals) if cosine_vals else None
            const_mean  = np.mean(const_vals)  if const_vals  else None

            if cosine_mean is not None or const_mean is not None:
                delta = (const_mean - cosine_mean) if (cosine_mean and const_mean) else None
                rows.append({
                    "dataset": ds,
                    "algo": algo,
                    "cosine_warmup": f"{cosine_mean:.4f}" if cosine_mean else "N/A",
                    "constant_lr1e-4": f"{const_mean:.4f}" if const_mean else "N/A",
                    "delta": f"{delta:+.4f}" if delta is not None else "N/A",
                    "winner": ("const" if delta > 0 else "cosine") if delta is not None else "N/A",
                })

    if not rows:
        print("没有找到任何结果，请先运行实验。")
    else:
        df = pd.DataFrame(rows)
        print(f"\n=== {metric} 对比（seeds={seeds}）===")
        print(df.to_string(index=False))

        # 按 algo 汇总 win/loss/tie
        print("\n--- 胜负汇总 ---")
        for algo in GNN_ALGOS:
            sub = df[df["algo"] == algo]
            wins   = (sub["winner"] == "const").sum()
            losses = (sub["winner"] == "cosine").sum()
            na     = (sub["winner"] == "N/A").sum()
            print(f"  {algo}: const_lr wins={wins}, cosine wins={losses}, N/A={na}")
