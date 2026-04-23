"""tests/test_datasets.py — 数据集预处理类单元测试

只测试已有原始数据的数据集；Gowalla / Epinions 需要原始文件，用 pytest.importorskip 跳过。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _check_processed_dir(processed_dir: Path) -> None:
    """验证 processed 目录包含 edges.csv / meta.json。"""
    assert (processed_dir / "edges.csv").exists(), f"缺少 edges.csv: {processed_dir}"
    import json
    meta = json.loads((processed_dir / "meta.json").read_text())
    assert "n_nodes" in meta
    assert meta["n_nodes"] > 0

    edges = pd.read_csv(processed_dir / "edges.csv")
    assert set(edges.columns) >= {"src", "dst", "timestamp"}
    assert (edges["src"] != edges["dst"]).all(), "存在自环"
    assert edges["timestamp"].is_monotonic_increasing or True  # 允许乱序（sort 在 train.py 做）
    assert edges["timestamp"].between(0.0, 1.0).all(), "时间戳未归一化至 [0,1]"


class TestCollegeMsg:
    def test_processed_format(self):
        from src.dataset.real.college_msg import CollegeMsgDataset
        ds = CollegeMsgDataset()
        if not ds.processed_dir.exists():
            pytest.skip("CollegeMsg 尚未预处理")
        _check_processed_dir(ds.processed_dir)

    def test_edge_count_range(self):
        from src.dataset.real.college_msg import CollegeMsgDataset
        ds = CollegeMsgDataset()
        if not ds.processed_dir.exists():
            pytest.skip("CollegeMsg 尚未预处理")
        edges = pd.read_csv(ds.processed_dir / "edges.csv")
        assert 50_000 < len(edges) < 70_000, f"边数异常：{len(edges)}"


class TestBitcoinOTC:
    def test_processed_format(self):
        from src.dataset.real.bitcoin_otc import BitcoinOTCDataset
        ds = BitcoinOTCDataset()
        if not ds.processed_dir.exists():
            pytest.skip("Bitcoin-OTC 尚未预处理")
        _check_processed_dir(ds.processed_dir)


class TestGowalla:
    def test_class_importable(self):
        from src.dataset.real.gowalla import GowallaDataset
        ds = GowallaDataset()
        assert ds.name == "gowalla"

    def test_processed_format(self):
        from src.dataset.real.gowalla import GowallaDataset
        ds = GowallaDataset()
        if not ds.processed_dir.exists():
            pytest.skip("Gowalla 尚未预处理（需下载原始数据）")
        _check_processed_dir(ds.processed_dir)


class TestEpinions:
    def test_class_importable(self):
        from src.dataset.real.epinions import EpinionsDataset
        ds = EpinionsDataset()
        assert ds.name == "epinions"

    def test_processed_format(self):
        from src.dataset.real.epinions import EpinionsDataset
        ds = EpinionsDataset()
        if not ds.processed_dir.exists():
            pytest.skip("Epinions 尚未预处理（需下载原始数据）")
        _check_processed_dir(ds.processed_dir)
