"""生成 two_hop_random 召回的网格实验 configs。

grid: top_k ∈ {10, 20} × total_rounds ∈ {50, 100, 200}
datasets: 小图 5 个 + 大图 4 个
rankers: gnn, random
命名: {dataset}_thr_{ranker}_k{top_k}_r{rounds}.yaml
"""
import os
from pathlib import Path

import yaml

SMALL = {
    "college_msg": "college_msg",
    "bitcoin_otc": "bitcoin_otc",
    "bitcoin_alpha": "bitcoin_alpha",
    "email_eu": "email_eu",
    "dnc_email": "dnc_email",
}
LARGE = {
    "sx_mathoverflow": "sx_mathoverflow",
    "sx_askubuntu": "sx_askubuntu",
    "sx_superuser": "sx_superuser",
    "epinions": "epinions",
}

TOP_K_LIST = [10, 20]
ROUNDS_LIST = [50, 100, 200]

OUT_DIR = Path("configs/online")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_cfg(dataset, size, ranker, top_k, total_rounds):
    is_large = size == "large"
    base = {
        "dataset": {
            "path": f"data/processed/{dataset}/edges.csv",
            "type": dataset,
        },
        "eval": {"degree_bins": 50, "graph_every_n": 10, "k_list": [1, 3, 5]},
        "feedback": {
            "cooldown_mode": "decay",
            "cooldown_rounds": 5,
            "p_neg": 0.02,
            "p_pos": 0.8,
        },
        "init_edge_ratio": 0.05,
        "init_strategy": "stratified",
        "recall": {"method": "two_hop_random", "top_k_recall": 100},
        "recommend": {"cold_start_random_fill": True, "top_k": top_k},
        "replay": {
            "capacity": 500 if is_large else 200,
            "sample_n": 64 if is_large else 32,
        },
        "runtime": {
            "device": "cpu" if ranker == "random" else "cuda",
            "log_every": 1,
            "out_dir": f"results/online/{dataset}_thr_{ranker}_k{top_k}_r{total_rounds}",
            "seed": 42,
        },
        "total_rounds": total_rounds,
        "trainer": {"update_every_n_rounds": 1},
        "user_selector": {
            "alpha": 0.5, "beta": 2.0, "gamma": 2.0, "lam": 0.1,
            "sample_ratio": 0.01 if is_large else 0.1,
            "strategy": "composite", "w": 3,
        },
    }

    if ranker == "gnn":
        base["model"] = {
            "encoder_type": "last", "hidden_dim": 8,
            "node_feat_dim": 0, "num_layers": 3, "type": "gnn",
        }
        base["trainer"] = {
            "batch_subgraph_max_hop": 2, "grad_clip": 1.0, "lr": 0.001,
            "max_neighbors": 30, "min_batch_size": 4,
            "scheduler": {
                "cycle_rounds": 25, "min_lr": 1e-5,
                "strategy": "cyclic", "warmup_rounds": 5,
            },
            "score_chunk_size": 1024 if is_large else 512,
            "update_every_n_rounds": 1,
            "use_amp": True,
        }
        base["replay"] = {
            "capacity": 500 if is_large else 200,
            "sample_n": 64 if is_large else 32,
        }
    else:
        base["model"] = {"type": "random"}
        base["replay"] = {"capacity": 0, "sample_n": 0}

    return base


configs_written = 0
for top_k in TOP_K_LIST:
    for total_rounds in ROUNDS_LIST:
        for ds, dtype in {**SMALL, **LARGE}.items():
            size = "large" if ds in LARGE else "small"
            for ranker in ["gnn", "random"]:
                cfg = make_cfg(ds, size, ranker, top_k, total_rounds)
                fname = OUT_DIR / f"{ds}_thr_{ranker}_k{top_k}_r{total_rounds}.yaml"
                with open(fname, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=True)
                configs_written += 1

print(f"Generated {configs_written} configs in {OUT_DIR}/")
