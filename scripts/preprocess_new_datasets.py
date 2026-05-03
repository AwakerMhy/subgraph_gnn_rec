"""scripts/preprocess_new_datasets.py — 批量预处理新增数据集

用法：
    python scripts/preprocess_new_datasets.py [--datasets all|facebook_ego|...|advogato|digg|higgs_reply]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOADERS = {
    "facebook_ego": ("src.dataset.real.facebook_ego", "FacebookEgoDataset"),
    "twitch_gamers": ("src.dataset.real.twitch_gamers", "TwitchGamersDataset"),
    "lastfm_asia": ("src.dataset.real.lastfm_asia", "LastFMAsiaDatset"),
    "epinions": ("src.dataset.real.epinions", "EpinionsDataset"),
    "ogbl_collab": ("src.dataset.real.ogbl_collab", "OgblCollabDataset"),
    "advogato": ("src.dataset.real.advogato", "AdvogatoDataset"),
    "digg": ("src.dataset.real.digg", "DiggDataset"),
    "higgs_reply": ("src.dataset.real.higgs_reply", "HiggsReplyDataset"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all"] + list(LOADERS.keys()),
    )
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--force", action="store_true", help="强制重新预处理")
    args = parser.parse_args()

    targets = list(LOADERS.keys()) if "all" in args.datasets else args.datasets

    ok, failed = [], []
    for name in targets:
        mod_path, cls_name = LOADERS[name]
        print(f"\n{'='*50}")
        print(f"预处理: {name}")
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            ds = cls(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
            ds.load(force_reprocess=args.force)
            print(f"  OK: {name} 完成")
            ok.append(name)
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}", file=sys.stderr)
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"完成: {ok}")
    if failed:
        print(f"失败: {failed}")


if __name__ == "__main__":
    main()
