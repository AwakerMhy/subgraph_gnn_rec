"""scripts/download_new_datasets.py — 批量下载新数据集原始文件

用法：
    python scripts/download_new_datasets.py [--datasets all|facebook_ego|twitch_gamers|lastfm_asia|epinions|ogbl_collab]
    python scripts/download_new_datasets.py --datasets facebook_ego lastfm_asia
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")

DATASETS = {
    "facebook_ego": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "dest_dir": RAW_DIR / "facebook_ego",
        "filename": "facebook_combined.txt.gz",
        "post": "gunzip",  # 解压 gz
    },
    "twitch_gamers": {
        "url": "https://snap.stanford.edu/data/twitch_gamers/large_twitch_edges.csv",
        "dest_dir": RAW_DIR / "twitch_gamers",
        "filename": "large_twitch_edges.csv",
        "post": None,
    },
    "lastfm_asia": {
        "url": "https://raw.githubusercontent.com/benedekrozemberczki/datasets/master/lastfm_asia/lastfm_asia_edges.csv",
        "dest_dir": RAW_DIR / "lastfm_asia",
        "filename": "lastfm_asia_edges.csv",
        "post": None,
    },
    "epinions": {
        "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
        "dest_dir": RAW_DIR / "epinions",
        "filename": "soc-Epinions1.txt.gz",
        "post": "gunzip",
    },
    "ogbl_collab": {
        "url": None,  # 通过 ogb 包自动下载
        "dest_dir": RAW_DIR / "ogbl_collab",
        "filename": None,
        "post": "ogb",
    },
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  下载: {url}")
    print(f"  →  {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            mb = count * block_size / 1e6
            print(f"\r  进度: {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress


def gunzip(gz_path: Path) -> Path:
    out_path = gz_path.with_suffix("")  # 去掉 .gz
    if out_path.exists():
        print(f"  已存在（跳过解压）: {out_path}")
        return out_path
    print(f"  解压: {gz_path} → {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def handle_ogb(dest_dir: Path) -> None:
    print("  ogbl-collab 将在预处理时由 ogb 包自动下载到:", dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import ogb  # noqa: F401
        print("  ogb 已安装 ✓")
    except ImportError:
        print("  ⚠ ogb 未安装，请先运行：pip install ogb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all"] + list(DATASETS.keys()),
        help="要下载的数据集（默认 all）",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    for name in targets:
        cfg = DATASETS[name]
        print(f"\n{'='*50}")
        print(f"数据集: {name}")
        dest_dir: Path = cfg["dest_dir"]

        if cfg["post"] == "ogb":
            handle_ogb(dest_dir)
            continue

        dest_file = dest_dir / cfg["filename"]

        # 检查目标文件是否已存在
        final_file = dest_file.with_suffix("") if cfg["post"] == "gunzip" else dest_file
        if final_file.exists():
            print(f"  已存在（跳过下载）: {final_file}")
            continue

        try:
            download_file(cfg["url"], dest_file)
        except Exception as e:
            print(f"  ✗ 下载失败: {e}", file=sys.stderr)
            print(f"  请手动下载: {cfg['url']}")
            print(f"  放置到: {dest_file}")
            continue

        if cfg["post"] == "gunzip":
            try:
                gunzip(dest_file)
            except Exception as e:
                print(f"  ✗ 解压失败: {e}", file=sys.stderr)

    print(f"\n{'='*50}")
    print("下载完成！运行预处理：")
    print("  python scripts/preprocess_new_datasets.py")


if __name__ == "__main__":
    main()
